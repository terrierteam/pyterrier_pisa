from typing import List

def pisa_artifact_metadata_adapter(path: str, dir_listing: List[str]):
    """
    Guess whether this path is a pisa index.
    pyterrier_pisa used to use pt_pisa_config.json instead of pt_meta.json. Use this file to assume they are pisa indexes.
    """
    if 'pt_pisa_config.json' in dir_listing:
        return {
            'type': 'sparse_index',
            'format': 'pisa',
            'package_hint': 'pyterrier-pisa',
        }
