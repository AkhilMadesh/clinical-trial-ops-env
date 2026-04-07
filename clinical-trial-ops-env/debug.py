import traceback
from env.environment import ClinicalTrialEnv

try:
    env = ClinicalTrialEnv(task_id='lab_dose_management')
    env.reset()
except Exception as e:
    with open('err.txt', 'w', encoding='utf-8') as f:
        traceback.print_exc(file=f)
