"""Runtime support for the frozen V9 controller."""

import copy

from element_source_guard_v9 import deficits

VERSION = 'explicit-heavy-element-multiplicity-v1'

def screen(metric):
    result = copy.deepcopy(metric)
    primary = result.get('primary_source_issues', result.get('source_issues', []))
    issues = [dict(step_id=s.get('step_id', str(i)), deficits=missing)
              for i, s in enumerate(result['route']['steps'])
              if (missing := deficits(s['product'], s['reactants']))]
    result['primary_source_issues'] = primary
    result['heavy_element_source_valid'] = not issues
    result['heavy_element_source_deficits'] = issues
    result['source_issues'] = [*primary, *issues]
    result['full_pass_after_source_screen'] = bool(result['full_pass'] and not result['source_issues'])
    result['source_audit_version'] = VERSION
    return result
