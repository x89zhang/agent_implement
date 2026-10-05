"""Read-only recomputation: run from repository root; optional output JSON path."""
import json,sys
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path.cwd()/'scripts'))
from hermes_monitor_rules import replay_sources,label
records=[];selected=[]
def phase_records(p,group,condition):
 if not (p/'evaluation.json').exists() or not (p/'defenses.json').exists():return
 ev=json.loads((p/'evaluation.json').read_text());man=json.loads((p/'defenses.json').read_text())
 for m,s in replay_sources(p,man).items():
  d=json.loads(s['path'].read_text()) if s['path'].exists() else None
  err='' if s['status'].get('status')=='completed' and d else 'replay incomplete/missing'
  lab=label(m,d,err)
  records.append(dict(group=group,condition=condition,method=m,phase=str(p),attack_success=ev.get('attack_success'),utility=ev.get('utility'),supplemented=s['supplemented'],**lab))
# Select the latest experiment for each task independently.
# Never fall back to an older paired batch when the latest batch is incomplete.
for root in sorted(Path('jobs').glob('agentdojo*')):
    base = root / 'Hermes/gpt/all_monitors'
    if not base.exists():
        continue
    names = {p.name for condition in ('skill_injection', 'no_injection')
             for p in (base / condition).glob('*batch')}
    if not names:
        continue
    name = max(names)
    for condition in ('skill_injection', 'no_injection'):
        batch = base / condition / name
        if not (batch / 'summary.json').exists():
            selected.append(dict(group=root.name, condition=condition, batch=str(batch), status='missing_summary'))
            continue
        summary = json.loads((batch / 'summary.json').read_text())
        evaluations = []
        for run in sorted(batch.glob('run_*')):
            phase = run / 'target'
            if (phase / 'evaluation.json').exists():
                evaluations.append(json.loads((phase / 'evaluation.json').read_text()))
            phase_records(phase, root.name, condition)
        selected.append(dict(group=root.name, condition=condition, batch=str(batch),
            status='available', model=summary['config']['agent.yaml']['llm']['model'],
            n=len(evaluations), as_count=sum(e.get('attack_success') is True for e in evaluations),
            utility_count=sum(e.get('utility') is True for e in evaluations),
            suite=evaluations[0]['suite'] if evaluations else None))
if not selected:
    raise SystemExit('No AgentDojo all_monitors batch found')
def table(recs):
 result=[]
 for m in dict.fromkeys(r['method'] for r in recs):
  rows=[r for r in recs if r['method']==m];on=[r for r in rows if r['condition'] in ['skill_injection','target']];off=[r for r in rows if r['condition'] in ['no_injection','control']]
  c={k:sum(r['status']=='alarm' for r in v) for k,v in [('on',on),('off',off)]}
  mat=Counter(('AS' if r['attack_success'] else 'AF')+'_'+r['status'] for r in on if isinstance(r['attack_success'],bool))
  result.append(dict(method=m,on_alarm=c['on'],on_valid=sum(r['status']!='error' for r in on),on_errors=sum(r['status']=='error' for r in on),off_alarm=c['off'],off_valid=sum(r['status']!='error' for r in off),off_errors=sum(r['status']=='error' for r in off),outcome=dict(mat),zero_decision=sum(r['decision_records']==0 for r in rows)))
 return result
out=dict(selected=selected,records=records,per_group={g:table([r for r in records if r['group']==g]) for g in dict.fromkeys(r['group'] for r in records)})
# Count actual ToolSafe pre-tool decisions in clean AgentDojo trajectories.
from hermes_monitor_rules import is_error, is_alarm
steps = Counter()
by_group = {}
for selected_batch in selected:
    if selected_batch.get('condition') != 'no_injection' or selected_batch.get('status') != 'available':
        continue
    counts = Counter()
    for phase in Path(selected_batch['batch']).glob('run_*/target'):
        manifest = json.loads((phase / 'defenses.json').read_text())
        source = replay_sources(phase, manifest).get('toolsafe')
        if not source or not source['path'].exists():
            continue
        defense = json.loads(source['path'].read_text())
        for event in defense['trace']:
            if event.get('step') != 'defense_before_tool':
                continue
            decision = (event.get('decisions') or {}).get('_last_toolsafe_decision')
            if not isinstance(decision, dict):
                continue
            counts['error' if is_error(decision) else 'alarm' if is_alarm('toolsafe', [decision]) else 'clean'] += 1
    by_group[selected_batch['group']] = dict(counts)
    steps.update(counts)
out['clean_pretool_decisions'] = {'toolsafe': {'per_group': by_group}}
output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/tmp/hermes_monitors_paper_comparison_recomputed.json')
output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
print('Output:', output)
print('SELECTED',json.dumps(selected,indent=2))
for g,t in out['per_group'].items():
 print('\n',g)
 for x in t:print(x['method'],f"{x['on_alarm']}/{x['on_valid']} {x['off_alarm']}/{x['off_valid']}",x['outcome'],'errors',x['on_errors'],x['off_errors'],'zero',x['zero_decision'])
