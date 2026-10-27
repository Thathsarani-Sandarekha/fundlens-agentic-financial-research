r"""FundLens: reproducible, read-only, Groq-judged answer-quality audit.

Run from the GitHub project root:
  set "PYTHONPATH=%CD%\src"
  python scripts\evaluate_quality_fast.py

Defaults to Run 3 for quality (5 questions / 25 retrieved passages), and aggregates
all three runs for deterministic repeatability. Does NOT run the agent, ingest,
or modify the DB. Requires the project's existing Groq credentials.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / 'evaluation' / 'repeatability'
DEFAULT_OUTPUT = ROOT / 'evaluation' / 'quality_auto'
MODEL_DEFAULT = 'openai/gpt-oss-20b'
CITE_RE = re.compile(r'\[([1-9]\d?)\]|【([1-9]\d?)(?:†[^】]*)?】')


def citations(answer: str) -> list[int]:
    return [int(a or b) for a, b in CITE_RE.findall(answer or '')]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    pos = (len(vals)-1)*q
    lower = int(pos)
    upper = min(lower+1,len(vals)-1)
    return round(vals[lower] + (pos-lower)*(vals[upper]-vals[lower]), 3)


def load_runs(folder: Path) -> dict[int, list[dict]]:
    result = {}
    for idx in (1, 2, 3):
        path = folder / f'run_{idx}_raw_results.json'
        if not path.exists():
            raise FileNotFoundError(f'Missing {path}. Place the six repeatability files in {folder}.')
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(data, list) or not data:
            raise ValueError(f'Expected a nonempty list in {path}')
        result[idx] = data
    cases = [[c['test_id'] for c in run] for run in result.values()]
    if not all(x == cases[0] for x in cases[1:]):
        raise ValueError('Runs have different ordered test IDs; cannot treat as matched repeats.')
    return result


def deterministic(runs: dict[int,list[dict]]) -> dict:
    all_rows = [case for run in runs.values() for case in run]
    numeric = [case for case in all_rows if case.get('expected_sql_value_usd_millions') is not None]
    rag_rows = [case for case in all_rows if case.get('retrieved_chunks')]
    by_id = {}
    for run in runs.values():
        for case in run:
            by_id.setdefault(case['test_id'], []).append(case)
    latency = [float(x['latency_seconds']) for x in all_rows]
    routes = {}
    for x in all_rows:
        name = x.get('expected_intent')
        routes.setdefault(name, []).append(x)
    def is_pass(c):
        return c.get('exception') is None and all(c.get('checks',{}).values())
    ref_valid = [bool(citations(x.get('final_answer',''))) and all(1 <= i <= len(x['retrieved_chunks']) for i in citations(x.get('final_answer',''))) for x in rag_rows]
    return {
        'runs':len(runs), 'unique_questions':len(by_id),'executions':len(all_rows),
        'automatic_pass':sum(is_pass(c) for c in all_rows),
        'routing_correct':sum(c.get('predicted_intent') == c.get('expected_intent') for c in all_rows),
        'route_consistency_questions':sum(len({c.get('predicted_intent') for c in rows})==1 for rows in by_id.values()),
        'sql_numeric_correct':sum(c.get('checks',{}).get('sql_exact_value') is True for c in numeric),
        'sql_numeric_checked':len(numeric),
        'nonempty_final_answers':sum(bool((c.get('final_answer') or '').strip()) for c in all_rows),
        'rag_or_hybrid_executions':len(rag_rows),
        'rag_cases_with_valid_citation_indices':sum(ref_valid),
        'retrieval_metadata_mismatch':sum(any(
            c.get('company_ticker') != 'GOOGL' or c.get('fiscal_year') != 2024 or
            c.get('fiscal_quarter') != 'FY' or c.get('section') != 'Risk Factors'
            for c in x['retrieved_chunks']) for x in rag_rows),
        'latency_seconds':{'mean':round(statistics.mean(latency),3), 'median':round(statistics.median(latency),3), 'p95':percentile(latency,.95), 'min':min(latency),'max':max(latency)},
        'latency_by_route':{k:{'count':len(v),'mean':round(statistics.mean(float(c['latency_seconds']) for c in v),3)} for k,v in routes.items()},
        'notes':['Automatic pass means automated benchmark checks, not factual faithfulness.',
                 'Citation index validity is NOT citation precision or PDF page accuracy.',
                 'All queries are curated GOOGL examples, not an independent generalization set.'],
    }


SYSTEM = """You are a STRICT independent financial research evaluator. Do not generate a new answer.
Assess ONLY the QUESTION, FINAL_ANSWER, five RETRIEVED_PASSAGES, and (for Hybrid) verified SQL rows.
Do NOT assume the retrieved passages are accurate representations of the original PDF; this is an evidence-grounded LLM judgment, not independently verified PDF fact checking.
Return ONLY a single valid JSON object. No markdown. Required keys:
- relevance_grades: exactly five integers each 0, 1, 2, or 3 matching passage indices [1]..[5]. 0=irrelevant, 1=tangential, 2=relevant, 3=direct evidence answering question.
- claims: list of atomic verifiable statements from FINAL_ANSWER, one per item, each {"text":str,"verdict":"supported"|"unsupported"|"contradicted","evidence_indices":list of passage indices}. A numeric fact may be supported by verified SQL rows instead of a passage; give empty evidence_indices then. Do not overlook partial or subtly unsupported statements.
- citation_checks: for EVERY DISTINCT numbered inline citation in FINAL_ANSWER, exactly one {"index":int,"supported":true|false}. A citation must directly support its attached claim(s), not just contain similar keywords. If the final answer has no citations, return empty list. Do not invent indices or cite unrelated text.
- answer_relevance_1to5: integer 1..5 for how directly the answer addresses the question.
- answer_completeness_1to5: integer 1..5 for coverage of key requested information based on evidence available.
- notes: brief string pointing out any important hallucination, reference ambiguity, or incomplete evidence.
Be conservative; if a claim lacks evidence mark unsupported. Output JSON ONLY."""


def judge_payload(case: dict) -> dict:
    passages = [{
        'index':i, 'page':c.get('page_number'), 'filing': f"{c.get('company_ticker')} {c.get('filing_type')} {c.get('fiscal_quarter')} {c.get('fiscal_year')}",
        'section':c.get('section'), 'text':c.get('content','')[:2300]
    } for i,c in enumerate(case.get('retrieved_chunks') or [],1)]
    return {
        'QUESTION':case['question'], 'FINAL_ANSWER':case.get('final_answer',''),
        'VERIFIED_SQL_ROWS':case.get('sql_rows') or [],
        'RETRIEVED_PASSAGES':passages,
        'DISTINCT_NUMBERED_INLINE_CITATIONS':sorted(set(citations(case.get('final_answer','')))),
    }


def parse_response(raw: str) -> dict:
    raw = (raw or '').strip()
    if raw.startswith('```'):
        raw = re.sub(r'^```(?:json)?\s*|\s*```$', '',raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        left,right = raw.find('{'),raw.rfind('}')
        if left >=0 and right >left:
            return json.loads(raw[left:right+1])
        raise


def validate_judgment(case: dict, result: dict) -> dict:
    n = len(case['retrieved_chunks'])
    grades = result['relevance_grades']
    if not isinstance(grades,list) or len(grades) != n or any(type(v) is not int or v not in (0,1,2,3) for v in grades):
        raise ValueError(f'Invalid relevance_grades length or values: {grades}')
    claims = result['claims']
    if not isinstance(claims,list) or not claims:
        raise ValueError('Judge must list atomic claims; got none')
    for c in claims:
        if c.get('verdict') not in ('supported','unsupported','contradicted') or not isinstance(c.get('text'),str):
            raise ValueError(f'Invalid claim: {c}')
        for idx in c.get('evidence_indices',[]):
            if type(idx) is not int or not 1 <= idx <= n:
                raise ValueError(f'Invalid claim evidence index: {idx}')
    cited = set(citations(case.get('final_answer','')))
    checks = result['citation_checks']
    seen = set()
    for ck in checks:
        idx = ck.get('index')
        if type(idx) is not int or idx not in cited or type(ck.get('supported')) is not bool or idx in seen:
            raise ValueError(f'Unexpected, duplicate or invalid citation check: {ck}')
        seen.add(idx)
    if seen != cited:
        raise ValueError(f'Citation checks mismatch: expected {cited}, got {seen}')
    for key in ('answer_relevance_1to5','answer_completeness_1to5'):
        if type(result.get(key)) is not int or not 1 <= result[key] <= 5:
            raise ValueError(f'Invalid {key}')
    return result


def call_judge(client, case:dict, model:str) -> dict:
    from fundlens.llm.groq_client import call_with_backoff
    content = json.dumps(judge_payload(case),ensure_ascii=False)
    errors = []
    for attempt in (1,2):
        try:
            args = dict(model=model,
                messages=[{'role':'system','content':SYSTEM}, {'role':'user','content':content}],
                temperature=0, max_tokens=2300)
            # Some models/endpoints do not support response_format; retry normally.
            if attempt==1:
                args['response_format']={'type':'json_object'}
            response = call_with_backoff(client, **args)
            raw = response.choices[0].message.content or ''
            return validate_judgment(case, parse_response(raw))
        except Exception as exc:
            errors.append(str(exc))
    raise RuntimeError('Judge returned unusable output twice: '+'; '.join(errors)[:900])


def dcg(values:list[int]) -> float:
    import math
    return sum((2**v-1)/math.log2(i+2) for i,v in enumerate(values))


def retrieval_scores(grades:list[int]):
    binary = [int(g>=2) for g in grades]
    prec = sum(binary)/len(binary)
    mrr = next((1/(i+1) for i,b in enumerate(binary) if b),0)
    ideal = dcg(sorted(grades,reverse=True))
    return prec,mrr,dcg(grades)/ideal if ideal else 0


def aggregate_judgments(data:dict) -> dict:
    scored = [v['result'] for v in data.values() if v.get('result')]
    if not scored:
        return {'status':'NO_JUDGMENTS','scored_cases':0}
    claims = [c for x in scored for c in x['claims']]
    checks = [c for x in scored for c in x['citation_checks']]
    ranks = [retrieval_scores(x['relevance_grades']) for x in scored]
    supported = sum(c['verdict']=='supported' for c in claims)
    return {'status':'COMPLETE' if len(scored)==len(data) else 'PARTIAL',
        'scored_cases':len(scored),'total_cases_requested':len(data),
        'claim_support_percent':round(supported/len(claims)*100,2) if claims else None,
        'supported_claims':supported,'all_claims':len(claims),
        'retrieval_precision_at_5_percent':round(statistics.mean(t[0] for t in ranks)*100,2),
        'retrieval_mrr_at_5':round(statistics.mean(t[1] for t in ranks),4),
        'retrieval_ndcg_at_5':round(statistics.mean(t[2] for t in ranks),4),
        'citation_support_percent':round(sum(c['supported'] for c in checks)/len(checks)*100,2) if checks else None,
        'supported_numbered_citations':sum(c['supported'] for c in checks),
        'all_numbered_citations':len(checks),
        'answer_relevance_mean_1to5':round(statistics.mean(x['answer_relevance_1to5'] for x in scored),2),
        'answer_completeness_mean_1to5':round(statistics.mean(x['answer_completeness_1to5'] for x in scored),2),
        'citation_pdf_page_accuracy':'NOT_VERIFIED_AGAINST_ORIGINAL_PDF',
        'limitations':'Custom LLM-as-a-judge against indexed chunks and SQL rows; not a ground-truth human or original-PDF audit.'}


def report(summary:dict,judgments:dict) -> str:
    d=summary['deterministic']; q=summary['llm_judge']
    lines=['# FundLens — automated evaluation','',
           f"Date: {summary['generated_at']} (UTC)",'',
           '## Deterministic 3-run benchmark','',
           '| Metric | Result |','|---|---:|',
           f"| Automatic case pass | {d['automatic_pass']}/{d['executions']} |",
           f"| Routing correct | {d['routing_correct']}/{d['executions']} |",
           f"| Routing consistency (repeated question) | {d['route_consistency_questions']}/{d['unique_questions']} |",
           f"| Exact SQL numeric matches | {d['sql_numeric_correct']}/{d['sql_numeric_checked']} |",
           f"| Nonempty final answers | {d['nonempty_final_answers']}/{d['executions']} |",
           f"| Valid numbered citation indices | {d['rag_cases_with_valid_citation_indices']}/{d['rag_or_hybrid_executions']} RAG/Hybrid cases |",
           f"| Aggregate latency p50 / p95 | {d['latency_seconds']['median']}s / {d['latency_seconds']['p95']}s |",
           '',
           '## Custom Groq LLM-as-a-judge — source-grounding assessment','',
           f"Model: `{summary['judge_model']}`. Quality basis: {summary['quality_basis']}",'',
           f"Status: **{q['status']}** ({q.get('scored_cases',0)}/{q.get('total_cases_requested',0)} judged cases)",'',
           '| Metric | Score |','|---|---:|']
    for key,title in [
        ('claim_support_percent','Claim support, evidence-grounded (%)'),
        ('retrieval_precision_at_5_percent','Retrieval Precision@5, grade >=2 (%)'),
        ('retrieval_mrr_at_5','Retrieval MRR@5'),
        ('retrieval_ndcg_at_5','Retrieval nDCG@5'),
        ('citation_support_percent','Numbered citation support (%)'),
        ('answer_relevance_mean_1to5','Mean answer relevance (1–5)'),
        ('answer_completeness_mean_1to5','Mean answer completeness (1–5)')]:
        lines.append(f"| {title} | {q.get(key,'—') if q.get(key) is not None else 'Not measured'} |")
    lines += ['', '**Original PDF citation page accuracy: not verified.**',
         '', '## Evaluation boundaries','',
         '- The deterministic 30/30 result is for 10 curated Alphabet questions, each tested 3 times. Not broad accuracy.',
         '- Groq judge scores are automated opinions based on retrieved passages, not independent human ground truth.',
         '- The scored RAG set is 5 questions / 25 chunks in Run 3 by default; use `--all-runs` to score 15 cases.',
         '- SQL numeric correctness is based on benchmark expected rows. Inline PDF citations for SQL facts are not yet implemented.',
         '- The benchmark previously showed occasional truncated *intermediate* Hybrid RAG text even though final answers were nonempty.',
         '- Metadata hit@5 must never be reported as Precision@5.', '',
         '## Per-case judge notes','']
    for key,j in judgments.items():
        if j.get('result'):
            x=j['result']; lines.append(f"- **{key}:** {x.get('notes','')} (grades: {x['relevance_grades']})")
        else:
            lines.append(f"- **{key}:** JUDGE FAILED: {j.get('error')}")
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=DEFAULT_INPUT)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--model',default=os.environ.get('FUNDLENS_JUDGE_MODEL',MODEL_DEFAULT))
    p.add_argument('--all-runs',action='store_true',help='Judge all 15 RAG/Hybrid cases, not just latest five')
    p.add_argument('--metrics-only',action='store_true',help='Skip the Groq judge; compute only deterministic metrics')
    p.add_argument('--force',action='store_true',help='Re-judge even when cached result exists')
    args=p.parse_args()
    runs=load_runs(args.input)
    deterministic_metrics=deterministic(runs)
    cases={f'run{r}_{c["test_id"]}':c for r in ((1,2,3) if args.all_runs else (3,)) for c in runs[r] if c.get('retrieved_chunks')}
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    cache_file=out/'judge_results.json'
    cache=json.loads(cache_file.read_text(encoding='utf-8')) if cache_file.exists() else {}
    judgments={}
    client=None
    for key,case in cases.items():
        fingerprint=hashlib.sha256(json.dumps(judge_payload(case),ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        cache_key=f'{args.model}|{fingerprint}'
        if cache_key in cache and not args.force:
            judgments[key]={'result':cache[cache_key]['result'], 'cached':True}
            print(f'  {key}: cached judgment')
            continue
        if args.metrics_only:
            continue
        if client is None:
            sys.path.insert(0,str(ROOT/'src'))
            try:
                from fundlens.llm.groq_client import get_raw_client
                client=get_raw_client()
            except Exception as exc:
                raise SystemExit('Cannot initialize FundLens Groq client. Activate docmind_env, ensure PYTHONPATH points to GitHub project src and GROQ_API_KEY is configured. Details: '+str(exc))
        try:
            result=call_judge(client,case,args.model)
            judgments[key]={'result':result,'cached':False}
            cache[cache_key]={'result':result,'test_id':case['test_id'],'model':args.model}
            cache_file.write_text(json.dumps(cache,indent=2,ensure_ascii=False),encoding='utf-8')
            print(f"  {key}: judged, grades={result['relevance_grades']}")
        except Exception as exc:
            judgments[key]={'result':None,'error':str(exc)}
            print(f'  {key}: judge FAILED: {exc}')
    if args.metrics_only:
        quality={'status':'SKIPPED','scored_cases':0,'total_cases_requested':len(cases)}
    else:
        # Ensure error rows are included in requested count.
        quality=aggregate_judgments(judgments)
    summary={'generated_at':datetime.now(timezone.utc).isoformat(timespec='seconds'),
             'judge_model':args.model,'quality_basis':'all 3 runs (15 cases)' if args.all_runs else 'Run 3 (5 RAG/Hybrid cases)',
             'deterministic':deterministic_metrics,'llm_judge':quality}
    (out/'quality_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'quality_report.md').write_text(report(summary,judgments),encoding='utf-8')
    (out/'judge_run_details.json').write_text(json.dumps(judgments,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nRESULT:',quality['status'], '| automatic passes:',f"{deterministic_metrics['automatic_pass']}/{deterministic_metrics['executions']}")
    print('Written:',out/'quality_summary.json','and',out/'quality_report.md')
    if not args.metrics_only and quality['status'] != 'COMPLETE':
        sys.exit(2)

if __name__=='__main__':
    main()
