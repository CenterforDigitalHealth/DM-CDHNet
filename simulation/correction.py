import json,types
from pathlib import Path
import torch
from torch.nn import functional as F
from token_roles import MED_MIN,N_MED
OUT=Path(__file__).resolve().parent

def install_correction(model,variant):
    if variant in ['baseline','gate_weighted_control','gate_unweighted']:return
    params=json.loads((OUT/'first_parameters.json').read_text());base=model.condition
    same_day=variant.endswith('same_day')
    def condition(self,hidden,gap,age,before,through):
        c=base(hidden,gap,age,before,through)
        tilted=c['joint_logp'].float().clone()
        tilted[...,0,3]+=params['bias_insulin'];tilted[...,0,9]+=params['bias_other']
        if same_day:
            # AGE is stored as integer days. TIME can be fractional before the
            # generator records int(AGE), so compare the prospective stored day.
            day=(age+gap).floor()
            for med in [3,9]:
                duplicate=(through[...,med]>=0)&(through[...,med]==day)
                tilted[..., :,med]=tilted[..., :,med].masked_fill(duplicate[...,None],-torch.inf)
        logmass=tilted.flatten(-2).logsumexp(-1)
        gate=c['gate'].float()+logmass
        joint=tilted-logmass[...,None,None]
        event=c['event_logp'].clone()
        event[..., self.nonmed_tokens]=event[..., self.nonmed_tokens]+F.logsigmoid(-gate)[...,None]-F.logsigmoid(-c['gate'].float())[...,None]
        event[...,MED_MIN:MED_MIN+N_MED]=F.logsigmoid(gate)[...,None]+joint.logsumexp(-2)
        c.update(gate=gate,joint_logp=joint,event_logp=event)
        return c
    model.condition=types.MethodType(condition,model)

