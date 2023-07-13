
import haystack.utils
from haystack.utils import clean_wiki_text
from haystack.utils import convert_files_to_docs
from haystack.utils import fetch_archive_from_http,print_answers
from haystack.document_stores import InMemoryDocumentStore
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.nodes import BM25Retriever
from haystack.nodes import FARMReader
from haystack.nodes import TransformersReader
from haystack.pipelines import ExtractiveQAPipeline
from config import TOP_K_READER,TOP_K_RETRIEVER,USE_GPU,HAYSTACK_READER

def getReaderResult(doc_dir,modelSelected,user_message):
    document_store = InMemoryDocumentStore(use_bm25=True)
    docs = convert_files_to_docs(dir_path=doc_dir,clean_func=clean_wiki_text,split_paragraphs=True)
    document_store.write_documents(docs)
    retriever = BM25Retriever(document_store=document_store)
    if HAYSTACK_READER == 'TRANSFORMER':
        reader = TransformersReader(model_name_or_path=modelSelected,tokenizer=modelSelected)
    else:
        reader = FARMReader(model_name_or_path=modelSelected, use_gpu=USE_GPU)
    
    pipe = ExtractiveQAPipeline(reader, retriever)
    results = pipe.run(query=user_message,params={"Retriever": {"top_k": TOP_K_RETRIEVER},"Reader": {"top_k": TOP_K_READER}})
    print('completed reader ')
    return results