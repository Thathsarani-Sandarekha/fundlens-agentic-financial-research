"""FundLens evidence-based evaluation, no external evaluation dependencies.

Run in project root:
  python scripts/evaluate_metrics.py run --suite core --repeat 2
  python scripts/evaluate_metrics.py score --results evaluation/results/<run-folder>

Checks only what can be verified with the returned agent state; manual PDF annotations
are required for faithfulness, semantic retrieval relevance and citation accuracy.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'src'))
LABELS = ('pure_sql', 'pure_rag', 'hybrid')
BASE_Q1 = 'https://s206.q4cdn.com/479360582/files/doc_financials/2025/q1/goog-10-q-q1-2025.pdf'
BASE_FY = 'https://s206.q4cdn.com/479360582/files/doc_financials/2024/q4/goog-10-k-2024.pdf'
SOURCES = {
    '2025_q1_income': {'url': BASE_Q1 + '#page=6', 'description': 'Alphabet 2025 Q1 10-Q, income statement; 2025 column (USD millions)'},
    '2024_fy_income': {'url': BASE_FY + '#page=54', 'description': 'Alphabet FY2024 10-K, income statement; 2024 column (USD millions)'},
    '2024_ai_competition': {'url': BASE_FY + '#page=12', 'description': 'Alphabet FY2024 10-K Risk Factors, AI competition/investment/infrastructure'},
    '2024_ai_liability': {'url': BASE_FY + '#page=17', 'description': 'Alphabet FY2024 10-K Risk Factors, AI liability/privacy/cybersecurity'},
}

# Pilot suite: values independently grounded in original filings. Expanded suite repeats
# facts with different question phrasings and splits AI-risk topics. Do not describe the
# expanded suite as 10 independent factual ground-truth observations.
TESTS = [
    dict(id='sql_q1_revenue', suite='core', intent='pure_sql', question="What was Alphabet's Q1 2025 consolidated revenue?", metric='revenue', expected=90234, source=['2025_q1_income']),
    dict(id='sql_fy_revenue', suite='core', intent='pure_sql', question="What was Alphabet's FY2024 revenue?", metric='revenue', expected=350018, source=['2024_fy_income']),
    dict(id='sql_fy_net_income', suite='core', intent='pure_sql', question="What was Alphabet's FY2024 net income?", metric='net_income', expected=100118, source=['2024_fy_income']),
    dict(id='rag_ai_risks', suite='core', intent='pure_rag', question='What AI-related risks did Alphabet discuss in its 2024 annual report?', ticker='GOOGL', year=2024, section='Risk Factors', source=['2024_ai_competition','2024_ai_liability']),
    dict(id='hybrid_revenue_risks', suite='core', intent='hybrid', question="What was Alphabet's FY2024 revenue, and what AI-related risks did its annual report discuss?", metric='revenue', expected=350018, ticker='GOOGL', year=2024, section='Risk Factors', source=['2024_fy_income','2024_ai_competition','2024_ai_liability']),
    dict(id='sql_q1_paraphrase', suite='expanded', intent='pure_sql', question='How much consolidated revenue did Alphabet generate in the first quarter of 2025?', metric='revenue', expected=90234, source=['2025_q1_income']),
    dict(id='sql_fy_paraphrase', suite='expanded', intent='pure_sql', question='Report the full-year 2024 total revenue for GOOGL.', metric='revenue', expected=350018, source=['2024_fy_income']),
    dict(id='rag_ai_infrastructure', suite='expanded', intent='pure_rag', question="What AI infrastructure, investment and patent risks did Alphabet mention in its 2024 10-K?", ticker='GOOGL', year=2024, section='Risk Factors', source=['2024_ai_competition']),
    dict(id='rag_ai_liability', suite='expanded', intent='pure_rag', question="Which risks of harmful AI content, privacy and cybersecurity did Alphabet identify in the 2024 annual report?", ticker='GOOGL', year=2024, section='Risk Factors', source=['2024_ai_liability']),
    dict(id='hybrid_net_income_risks', suite='expanded', intent='hybrid', question="What was Alphabet's FY2024 net income, and what AI-related risks did its annual report discuss?", metric='net_income', expected=100118, ticker='GOOGL', year=2024, section='Risk Factors', source=['2024_fy_income','2024_ai_liability']),
]


def exact_numbers(rows: Any, metric: str) -> list[Decimal]:
    values = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        for k, v in row.items():
            if str(k).lower() != metric or v is None:
                continue
            try:
                values.append(Decimal(str(v).replace(',', '')))
            except (InvalidOperation, ValueError):
                pass
    return values


def checks_for(test: dict, state: dict) -> dict:
    kind = test['intent']
    checks = {'intent_correct': state.get('intent') == kind,
              'answer_nonempty': bool(str(state.get('final_answer') or '').strip())}
    if kind in ('pure_sql','hybrid'):
        checks['sql_success'] = state.get('sql_success') is True
        checks['sql_exact_value'] = Decimal(test['expected']) in exact_numbers(state.get('sql_rows'), test['metric'])
    if kind in ('pure_rag','hybrid'):
        chunks = state.get('rag_chunks') or []
        checks['rag_success'] = state.get('rag_success') is True
        checks['rag_evidence_retrieved'] = bool(chunks)
        checks['rag_citations_present'] = bool(state.get('citations'))
        # Important: filing/section metadata match is NOT semantic relevance.
        checks['rag_metadata_hit_at_5'] = any(
            str(ch.get('company_ticker','')).upper() == test['ticker']
            and str(ch.get('fiscal_year','')) == str(test['year'])
            and str(ch.get('section','')).lower() == test['section'].lower()
            for ch in chunks[:5]
        )
    if kind == 'hybrid':
        checks['hybrid_combined'] = 'synth[combined]' in str(state.get('route_taken',''))
    return checks


def pct(numerator: int | float, denominator: int) -> float | None:
    return round(100*numerator/denominator,2) if denominator else None


def percentile(vals: list[float], quantile: float) -> float | None:
    if not vals: return None
    a = sorted(vals)
    rank = quantile*(len(a)-1)
    lo,hi=math.floor(rank),math.ceil(rank)
    return round(a[lo]+(a[hi]-a[lo])*(rank-lo),2)


def macro_scores(records: list[dict]) -> dict:
    per_label = {}
    for label in LABELS:
        tp=sum(r['expected_intent']==label and r['predicted_intent']==label for r in records)
        fp=sum(r['expected_intent']!=label and r['predicted_intent']==label for r in records)
        fn=sum(r['expected_intent']==label and r['predicted_intent']!=label for r in records)
        precision=tp/(tp+fp) if tp+fp else 0.0
        recall=tp/(tp+fn) if tp+fn else 0.0
        f1=(2*precision*recall/(precision+recall)) if precision+recall else 0.0
        per_label[label]={'precision':round(precision,4),'recall':round(recall,4),'f1':round(f1,4),'support':tp+fn}
    represented=[x for x in per_label.values() if x['support']>0]
    return {'macro_precision':round(statistics.mean(x['precision'] for x in represented),4) if represented else None,
            'macro_recall':round(statistics.mean(x['recall'] for x in represented),4) if represented else None,
            'macro_f1':round(statistics.mean(x['f1'] for x in represented),4) if represented else None,
            'macro_f1_class_count':len(represented), 'per_class':per_label}


def load_csv(path: Path) -> list[dict]:
    if not path.exists(): return []
    with path.open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore')
        w.writeheader();w.writerows(rows)


def valid_int(x: Any) -> int | None:
    if str(x).strip()=='': return None
    try:
        f=float(x)
        if not math.isfinite(f) or f<0 or not f.is_integer(): return None
        return int(f)
    except (ValueError,TypeError): return None


def valid_scale(x: Any) -> int | None:
    v=valid_int(x)
    return v if v is not None and 1<=v<=5 else None


def make_review_files(folder: Path, records: list[dict]):
    case_path=folder/'case_review.csv'
    rank_path=folder/'retrieval_review.csv'
    case_cols=['test_id','run','answer_relevance_1to5','answer_completeness_1to5',
               'claims_supported','claims_unsupported','claims_contradicted',
               'citations_checked','citations_correct','pages_checked','pages_correct',
               'reviewer_notes']
    rank_cols=['test_id','run','rank','chunk_id','section','page_number',
               'semantic_relevance_0to3','reviewer_notes']
    if not case_path.exists():
        cases=[]
        for r in records:
            cases.append({'test_id':r['test_id'],'run':r['run']})
        write_csv(case_path,cases,case_cols)
    if not rank_path.exists():
        chunks=[]
        for r in records:
            if r['expected_intent'] not in ('pure_rag','hybrid'): continue
            for i,c in enumerate(r['retrieved_chunks'][:5],1):
                chunks.append({'test_id':r['test_id'],'run':r['run'],'rank':i,
                               'chunk_id':str(c.get('chunk_id','')),
                               'section':str(c.get('section','')),'page_number':str(c.get('page_number',''))})
        write_csv(rank_path,chunks,rank_cols)
    return case_path,rank_path


def manual_metrics(folder: Path) -> dict:
    cases=load_csv(folder/'case_review.csv')
    chunks=load_csv(folder/'retrieval_review.csv')
    result={}
    for field,key in [('answer_relevance_1to5','answer_relevance_mean_1to5'),('answer_completeness_1to5','answer_completeness_mean_1to5')]:
        scores=[valid_scale(r.get(field)) for r in cases]
        valid=[x for x in scores if x is not None]
        result[key]=round(statistics.mean(valid),2) if valid else None
        result[key+'_rated_cases']=len(valid)
    ratios=[('claims_supported','claims_unsupported','claims_contradicted','faithfulness_claim_support_percent'),
            ('citations_correct','citations_checked',None,'citation_precision_percent'),
            ('pages_correct','pages_checked',None,'citation_page_accuracy_percent')]
    for good,denom,bad,key in ratios:
        observed=[r for r in cases if valid_int(r.get(good)) is not None and valid_int(r.get(denom)) is not None and (bad is None or valid_int(r.get(bad)) is not None)]
        if bad:
            nums=sum(valid_int(r[good]) for r in observed)
            total=sum(valid_int(r[good])+valid_int(r[denom])+valid_int(r[bad]) for r in observed)
        else:
            observed=[r for r in observed if valid_int(r[good])<=valid_int(r[denom])]
            nums=sum(valid_int(r[good]) for r in observed)
            total=sum(valid_int(r[denom]) for r in observed)
        result[key]=pct(nums,total)
        result[key+'_reviewed_items']=total
    by_case=defaultdict(list)
    for row in chunks:
        grade=valid_int(row.get('semantic_relevance_0to3'))
        if grade is None or grade>3: continue
        by_case[(row.get('test_id'),str(row.get('run')))].append((int(row['rank']),grade))
    prec,mrr,ndcg=[],[],[]
    all_records={(x['test_id'],str(x['run'])): x for x in load_json(folder/'raw_results.json')}
    for key,vals in by_case.items():
        vals.sort()
        # Score only cases with every returned top-5 chunk annotated.
        # A 3-chunk retrieval needs 3 labels; report both total and scored case counts.
        # Incomplete judgments are withheld from official metrics.
        record=all_records.get(key)
        if not record or len(vals)!=min(5,len(record['retrieved_chunks'])): continue
        grades=[grade for _,grade in vals]
        relevant=[i for i,g in enumerate(grades) if g>=1]
        prec.append(len(relevant)/len(grades) if grades else 0)
        mrr.append(1/(relevant[0]+1) if relevant else 0)
        gains=[(2**g-1)/(math.log2(i+2)) for i,g in enumerate(grades)]
        ideal=sorted(grades,reverse=True)
        best=sum((2**g-1)/math.log2(i+2) for i,g in enumerate(ideal))
        ndcg.append(sum(gains)/best if best else 0)
    result['retrieval_precision_at_5_percent']=round(100*statistics.mean(prec),2) if prec else None
    result['retrieval_mrr_at_5']=round(statistics.mean(mrr),4) if mrr else None
    result['retrieval_ndcg_at_5']=round(statistics.mean(ndcg),4) if ndcg else None
    result['retrieval_fully_labeled_cases']=len(prec)
    result['retrieval_recall_at_5']='NOT_MEASURABLE_WITHOUT_EXHAUSTIVE_GOLD_CHUNKS'
    result['manual_review_cases_total']=len(cases)
    return result


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def summary(records: list[dict], folder: Path) -> dict:
    sql_cases=[r for r in records if r['expected_intent'] in ('pure_sql','hybrid')]
    rag_cases=[r for r in records if r['expected_intent'] in ('pure_rag','hybrid')]
    hybrids=[r for r in records if r['expected_intent']=='hybrid']
    successful=[r for r in records if r['checks'].get('answer_nonempty') and not r.get('exception')]
    lat=[r['latency_seconds'] for r in records]
    per_id=defaultdict(list)
    for r in records: per_id[r['test_id']].append(r)
    mult=[v for v in per_id.values() if len(v)>1]
    result={
        'benchmark_size':len(records), 'unique_questions':len(per_id),
        'routing_accuracy_percent':pct(sum(r['checks'].get('intent_correct',False) for r in records),len(records)),
        **macro_scores(records),
        'sql_execution_success_percent':pct(sum(r['checks'].get('sql_success',False) for r in sql_cases),len(sql_cases)),
        'sql_exact_numeric_accuracy_percent':pct(sum(r['checks'].get('sql_exact_value',False) for r in sql_cases),len(sql_cases)),
        'rag_execution_success_percent':pct(sum(r['checks'].get('rag_success',False) for r in rag_cases),len(rag_cases)),
        'rag_metadata_hit_at_5_percent':pct(sum(r['checks'].get('rag_metadata_hit_at_5',False) for r in rag_cases),len(rag_cases)),
        'hybrid_full_success_percent':pct(sum(r['checks'].get('hybrid_combined',False) and r['checks'].get('sql_exact_value',False) and r['checks'].get('rag_evidence_retrieved',False) for r in hybrids),len(hybrids)),
        'automatic_case_pass_rate_percent':pct(sum(all(r['checks'].values()) for r in records),len(records)),
        'runtime_completion_percent':pct(len(successful),len(records)),
        'latency_seconds':{'mean':round(statistics.mean(lat),2) if lat else None,'p50':percentile(lat,.5),'p95':percentile(lat,.95),'min':min(lat) if lat else None,'max':max(lat) if lat else None},
        'repeatability':{'questions_repeated':len(mult),
                         'route_consistency_percent':pct(sum(len({r['predicted_intent'] for r in group})==1 for group in mult),len(mult)),
                         'numeric_accuracy_consistency_percent':pct(sum(len({r['checks'].get('sql_exact_value') for r in group})==1 for group in mult if group[0]['expected_intent'] in ('pure_sql','hybrid')),
                                                                    sum(1 for group in mult if group[0]['expected_intent'] in ('pure_sql','hybrid')))},
        'token_usage':'NOT_INSTRUMENTED', 'inference_cost_usd':'NOT_INSTRUMENTED',
        'human_evaluated_metrics':manual_metrics(folder),
        'limitations':['A five-case core suite is a smoke test, not a statistically significant benchmark.',
                       'Metadata retrieval hit is not semantic retrieval precision.',
                       'Rounded natural-language answers are not used for exact numeric comparisons.',
                       'LLM faithfulness/citation quality require verified PDF annotations.',
                       'Local document page labels may differ from PDF viewer page indexes.',
                       'API token counts and costs are not exposed by the current agent state.']
    }
    return result


def fmt(v: Any, suffix: str='') -> str:
    return 'Pending review / N/A' if v is None else str(v)+suffix


def save_report(folder: Path) -> None:
    records=load_json(folder/'raw_results.json')
    result=summary(records,folder)
    (folder/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lat=result['latency_seconds']; man=result['human_evaluated_metrics']
    lines=['# FundLens — system evaluation metrics','',
           '**Status:** Measured locally using retrieved agent state; human-source-reviewed metrics remain blank until annotated.','',
           f"- **Samples:** {result['benchmark_size']} runs across {result['unique_questions']} questions",
           f"- **Routing accuracy:** {fmt(result['routing_accuracy_percent'],'%')}",
           f"- **Routing macro-F1:** {fmt(result['macro_f1'])} (across {result['macro_f1_class_count']} represented classes)",
           f"- **SQL execution success:** {fmt(result['sql_execution_success_percent'],'%')}",
           f"- **Exact SQL value accuracy:** {fmt(result['sql_exact_numeric_accuracy_percent'],'%')}",
           f"- **RAG execution success:** {fmt(result['rag_execution_success_percent'],'%')}",
           f"- **Metadata hit@5 (NOT retrieval relevance):** {fmt(result['rag_metadata_hit_at_5_percent'],'%')}",
           f"- **Hybrid end-to-end success:** {fmt(result['hybrid_full_success_percent'],'%')}",
           f"- **All automatic checks passed:** {fmt(result['automatic_case_pass_rate_percent'],'%')}",
           f"- **Latency p50 / p95:** {fmt(lat['p50'],'s')} / {fmt(lat['p95'],'s')}",
           '', '## Human-evaluated quality metrics (verify against actual PDFs)','',
           '| Metric | Score | Status |','|---|---:|---|',
           f"| Faithfulness: supported claims / all reviewed claims | {fmt(man['faithfulness_claim_support_percent'],'%')} | {man['faithfulness_claim_support_percent_reviewed_items']} reviewed claims |",
           f"| Answer relevance (1–5) | {fmt(man['answer_relevance_mean_1to5'])} | {man['answer_relevance_mean_1to5_rated_cases']} reviewed cases |",
           f"| Answer completeness (1–5) | {fmt(man['answer_completeness_mean_1to5'])} | {man['answer_completeness_mean_1to5_rated_cases']} reviewed cases |",
           f"| Citation precision | {fmt(man['citation_precision_percent'],'%')} | {man['citation_precision_percent_reviewed_items']} reviewed citations |",
           f"| Citation PDF page accuracy | {fmt(man['citation_page_accuracy_percent'],'%')} | {man['citation_page_accuracy_percent_reviewed_items']} reviewed pages |",
           f"| Retrieval Precision@5 | {fmt(man['retrieval_precision_at_5_percent'],'%')} | {man['retrieval_fully_labeled_cases']} labeled RAG runs |",
           f"| Retrieval MRR@5 | {fmt(man['retrieval_mrr_at_5'])} | Top-ranked relevant evidence |",
           f"| Retrieval nDCG@5 | {fmt(man['retrieval_ndcg_at_5'])} | Graded relevance (0–3) |",
           '| Retrieval Recall@5 | N/A | Requires a complete set of relevant chunks per question |',
           '', '## Per-case automatic results','',
           '| Case | Route | SQL exact | RAG evidence | Full auto pass | Latency |',
           '|---|---|---|---|---|---:|']
    for r in records:
        c=r['checks']; b=lambda k: ('PASS' if c.get(k) else 'FAIL') if k in c else '—'
        lines.append(f"| `{r['test_id']}` #{r['run']} | `{r['predicted_intent']}` | {b('sql_exact_value')} | {b('rag_evidence_retrieved')} | {'PASS' if all(c.values()) else 'FAIL'} | {r['latency_seconds']:.2f}s |")
    lines.extend(['','## How to use the source evidence','',
                  'Open `evaluation/reference_sources.html` for direct PDF links. The app\'s chunk page number might not equal the original PDF printed page.',
                  'Do not report citation page accuracy until compared against the exact ingested PDF edition.',
                  '', '## Limitations and interpretation','']+[f'- {x}' for x in result['limitations']])
    (folder/'metrics_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    from metrics_dashboard import generate
    generate(folder, records, result)
    print('\nMetrics report:',folder/'metrics_report.md')
    print('Machine-readable:',folder/'summary.json')


def run(args) -> int:
    try:
        from fundlens.agent.router.graph import ask
    except Exception as ex:
        print('Failed to import FundLens. Activate docmind_env and run from the project root. Details:',ex,file=sys.stderr)
        return 2
    selection=[t for t in TESTS if args.suite=='all' or t['suite']=='core']
    if args.only:
        selection=[t for t in TESTS if t['id']==args.only]
    folder=ROOT/'evaluation'/'results'/datetime.now().strftime('metrics_%Y%m%d_%H%M%S')
    folder.mkdir(parents=True,exist_ok=True)
    out=[]
    for test in selection:
        for n in range(1,args.repeat+1):
            print(f"\n[{test['id']} {n}/{args.repeat}] {test['question']}",flush=True)
            started=time.perf_counter()
            try:
                state=ask(test['question'])
                checks=checks_for(test,state)
                err=None
            except Exception as ex:
                state={};checks={'request_completed':False};err=f'{type(ex).__name__}: {ex}'
            secs=round(time.perf_counter()-started,3)
            record={'test_id':test['id'],'run':n,'suite':test['suite'],
                    'question':test['question'],'expected_intent':test['intent'],
                    'predicted_intent':state.get('intent'),'intent_confidence':state.get('intent_confidence'),
                    'source_references':[SOURCES[x] for x in test['source']],
                    'expected_sql_metric':test.get('metric'),'expected_sql_value_usd_millions':test.get('expected'),
                    'checks':checks,'latency_seconds':secs,'exception':err,
                    'route':state.get('route_taken'),'sql_query':state.get('sql_query'),
                    'sql_error':state.get('sql_error'),'sql_rows':state.get('sql_rows'),
                    'rag_error':state.get('rag_error'),'rag_answer':state.get('rag_answer'),
                    'final_answer':state.get('final_answer'),
                    'retrieved_chunks':(state.get('rag_chunks') or [])[:5],
                    'citations':state.get('citations') or []}
            out.append(record)
            print(f"  {'PASS' if all(checks.values()) else 'FAIL'} | {secs:.1f}s | {record['route']} | {', '.join(k for k,v in checks.items() if not v)}",flush=True)
    (folder/'raw_results.json').write_text(json.dumps(out,indent=2,default=str),encoding='utf-8')
    case_path,rank_path=make_review_files(folder,out)
    save_report(folder)
    print('Review forms:',case_path,'and',rank_path)
    print('Source viewer:',ROOT/'evaluation'/'reference_sources.html')
    return 0 if all(all(r['checks'].values()) for r in out) else 1


def main():
    ap=argparse.ArgumentParser(description='FundLens evaluation with objective + human-review metrics')
    subs=ap.add_subparsers(dest='command',required=True)
    cmd=subs.add_parser('run',help='Call the real FundLens agent and create metrics/review sheets')
    cmd.add_argument('--suite',choices=['core','all'],default='core')
    cmd.add_argument('--repeat',type=int,choices=(1,2,3),default=1)
    cmd.add_argument('--only',choices=[x['id'] for x in TESTS])
    score=subs.add_parser('score',help='Recalculate report from completed human review CSV sheets')
    score.add_argument('--results',type=Path,required=True)
    args=ap.parse_args()
    if args.command=='run': return run(args)
    if not (args.results/'raw_results.json').exists():
        print('No raw_results.json in specified folder',file=sys.stderr);return 2
    make_review_files(args.results,load_json(args.results/'raw_results.json'))
    save_report(args.results)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
