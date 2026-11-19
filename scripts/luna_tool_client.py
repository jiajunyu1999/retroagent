"""Runtime support for the frozen V9 controller."""

import json

def adapt(raw, target_id, kind):
    raw = raw.strip()
    if raw.startswith('```'):
        raw = raw.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    decoder = json.JSONDecoder()
    values = []
    while raw:
        value, end = decoder.raw_decode(raw)
        values.append(value)
        raw = raw[end:].strip()
    if not values:
        raise ValueError('Empty JSON')
    if kind == 'route':
        if len(values) != 1 or not isinstance(values[0], dict):
            raise ValueError('Expected one route')
        result = dict(values[0])
        if result.get('target_id', target_id) != target_id or not isinstance(result.get('steps'), list):
            raise ValueError('Wrong target or route schema')
        result['target_id'] = target_id
        return result
    actions = []
    for value in values:
        if isinstance(value, list):
            objects = value
        elif isinstance(value, dict):
            if value.get('target_id', target_id) != target_id:
                raise ValueError('Conflicting target ID')
            objects = value['actions'] if isinstance(value.get('actions'), list) else [value]
        else:
            raise ValueError('Expected action object or list')
        for item in objects:
            if not isinstance(item, dict) or item.get('op') not in {'search', 'stock', 'propose', 'finish'}:
                raise ValueError('Unknown action')
            if item.get('target_id', target_id) != target_id:
                raise ValueError('Conflicting action target')
            actions.append(item)
    return {'target_id': target_id, 'actions': actions}
