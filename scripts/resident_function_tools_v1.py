"""Runtime support for the frozen V9 controller."""

import concurrent.futures as futures
import copy

import json

from pathlib import Path

import uuid

from multiprocessing.managers import BaseManager

PROVIDERS = ('LocalRetro', 'RetroChimeraEdit', 'AiZynthFinder',
             'Graph2Edits', 'RetroChimeraDeNovo')

def schema(name, description, properties, required):
    return dict(type='function', name=name, description=description,
                parameters=dict(type='object', properties=properties,
                                required=required, additionalProperties=False))

def strings(maximum):
    return dict(type='array', items=dict(type='string'), minItems=1, maxItems=maximum)

TOOLS = [
    schema('predict_reactions', 'Predict precursor reactions for exact product SMILES. '
           'Top10 per product; proposals require separate validation.',
           dict(provider=dict(type='string', enum=list(PROVIDERS)), products=strings(16)),
           ['provider', 'products']),
    schema('score_reactions', 'Original ChemCensor scoring, not a synthesis guarantee.',
           dict(reactions=strings(64)), ['reactions']),
    schema('check_stock', 'Exact fixed-stock membership; do not substitute similar molecules.',
           dict(smiles=strings(256)), ['smiles']),
    schema('find_materials', 'Similar structures in the pinned train-derived stock index. '
           'A neighbor is not an interchangeable precursor.',
           dict(smiles=dict(type='string'), limit=dict(type='integer', minimum=1, maximum=8)),
           ['smiles', 'limit']),
    schema('retrieve_reactions', 'Read-only exact-product training precedents. '
           'Evaluation must separately exclude direct target answers.',
           dict(product=dict(type='string'), limit=dict(type='integer', minimum=1, maximum=8)),
           ['product', 'limit']),
]

class Manager(BaseManager):
    pass

class Client:
    def __init__(self, out):
        spec = json.loads((Path(out) / 'ready.json').read_text())
        Manager.register('tools')
        self.manager = Manager(address=spec['socket'],
                               authkey=Path(spec['key_file']).read_bytes())
        self.manager.connect()
        self.remote = self.manager.tools()
        self.session = self.remote.open_session()

    def call(self, name, arguments, call_id=None):
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        return self.remote.call(self.session, call_id or uuid.uuid4().hex, name, arguments)

    def dispatch(self, function_call):
        """Consume only a named function and its JSON arguments, never code."""
        return self.call(function_call['name'], function_call['arguments'])

    def batch(self, calls, workers=5):
        if not 1 <= workers <= 8:
            raise ValueError('workers must be1..8')
        with futures.ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self.dispatch, calls))


def validate_call(name, arguments):
    definition = next((x for x in TOOLS if x['name'] == name), None)
    if definition is None:
        raise ValueError('Unknown tool')
    parameters = definition['parameters']
    if not isinstance(arguments, dict) or set(arguments) != set(parameters['required']):
        raise ValueError('Arguments must match required properties exactly')
    for key, spec in parameters['properties'].items():
        value = arguments[key]
        if spec['type'] == 'array':
            if (not isinstance(value, list)
                    or not spec['minItems'] <= len(value) <= spec['maxItems']
                    or not all(isinstance(x, str) and 0 < len(x) <= 16000 for x in value)):
                raise ValueError('Invalid molecule/reaction batch')
        elif spec['type'] == 'string':
            if not isinstance(value, str) or not 0 < len(value) <= 16000:
                raise ValueError('Expected nonempty string')
        elif spec['type'] == 'integer':
            if type(value) is not int or not spec['minimum'] <= value <= spec['maximum']:
                raise ValueError('Invalid limit')
        if 'enum' in spec and value not in spec['enum']:
            raise ValueError('Unknown provider')
    return copy.deepcopy(arguments)
