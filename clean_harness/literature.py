"""Real public metadata/abstract searches. Suggestions remain unverified hypotheses."""
import hashlib
import html
import json
import re
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

FAMILIES = ('amide coupling','esterification','Suzuki coupling','Buchwald Hartwig amination',
            'nucleophilic aromatic substitution','alkylation','reductive amination','oxidation',
            'reduction','hydrolysis','deprotection','heterocycle synthesis','cyclization',
            'carbonyl addition','olefination','ether synthesis','cross coupling')
MOTIFS = ('aliphatic','aromatic','heteroaromatic','amine','amide','ester','alcohol','phenol',
          'carboxylic acid','aryl halide','alkyl halide','nitrile','ketone','aldehyde','ether',
          'pyridine','pyrimidine','indole','piperidine','sulfonamide','thiophene','stereoselective')


def query_terms(request):
    family, motifs = request.get('family'), request.get('motifs',[])
    if family not in FAMILIES or not isinstance(motifs,list) or len(motifs)>4 or any(m not in MOTIFS for m in motifs):
        raise ValueError('Use generic allowed reaction families/motifs; no molecule IDs, names or exact SMILES queries')
    return family+' '+ ' '.join(motifs)+' organic synthesis'


def plain(value):
    return html.unescape(re.sub('<[^>]+>',' ',value or '')).strip()


class Literature:
    def __init__(self, cutoff, fetch=None):
        self.cutoff, self.fetch = cutoff, fetch or self._fetch
        self.sources, self.log = {}, []

    @staticmethod
    def _fetch(url, timeout):
        req = Request(url,headers={'User-Agent':'RetroAgent-literature/1.0 (academic retrosynthesis research)',
                                   'Accept':'application/json'})
        with urlopen(req,timeout=timeout) as reply:
            raw = reply.read(4_000_001)
            if len(raw)>4_000_000:
                raise ValueError('Oversized metadata response')
            return json.loads(raw)

    def search(self, request, provider, timeout=10.):
        q = query_terms(request)
        if provider == 'crossref':
            url = 'https://api.crossref.org/works?'+urlencode(dict(query=q,rows=5,
                filter='type:journal-article,until-pub-date:'+self.cutoff))
        elif provider == 'europepmc':
            url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?'+urlencode(dict(
                query=q+' FIRST_PDATE:[1900-01-01 TO '+self.cutoff+']',format='json',resultType='core',pageSize=5))
        else:
            raise ValueError('Unknown literature provider')
        started = time.time()
        record = dict(query=q,provider=provider,url=url,started=started,cutoff=self.cutoff)
        try:
            raw = self.fetch(url,timeout)
            record['response_sha256'] = hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest()
            rows = raw['message']['items'] if provider == 'crossref' else raw.get('resultList',{}).get('result',[])
            result = []
            for row in rows:
                doi = row.get('DOI',row.get('doi',''))
                title = row.get('title','')
                title = ' '.join(title) if isinstance(title,list) else title
                abstract = plain(row.get('abstract',row.get('abstractText','')))[:5000]
                link = ('https://doi.org/'+doi) if doi else 'https://europepmc.org/article/'+row.get('source','MED')+'/'+row.get('id','')
                sid = hashlib.sha256(link.encode()).hexdigest()[:16]
                source = dict(source_id=sid,title=plain(title),doi=doi,url=link,abstract=abstract,
                              provider=provider,query=q,cutoff=self.cutoff,
                              evidence_level='abstract' if abstract else 'metadata_only')
                self.sources[sid] = source
                result.append(source)
            record.update(status='completed',sources=result,raw_response=raw)
            return result
        except Exception as exc:
            record.update(status='failed',error=type(exc).__name__+': '+str(exc))
            return []
        finally:
            record['seconds'] = time.time()-started
            self.log.append(record)

    def validate_suggestion(self, row):
        ids, support = row.get('source_ids',[]),row.get('support',[])
        if not isinstance(ids,list) or not ids or any(s not in self.sources for s in ids):
            raise ValueError('Unknown or missing literature source')
        if not isinstance(support,list) or not support:
            raise ValueError('Need an actual supporting abstract excerpt')
        valid = set()
        for item in support:
            source = self.sources.get(item.get('source_id'))
            quote = item.get('quote','')
            if source and source['source_id'] in ids and 20 <= len(quote) <= 300 and quote in source['abstract']:
                valid.add(source['source_id'])
        if set(ids)-valid:
            raise ValueError('Every cited source needs a verbatim abstract excerpt; metadata is insufficient')
        return ids
