from haystack.document_stores import InMemoryDocumentStore
document_store = InMemoryDocumentStore(embedding_dim=512)
from haystack.utils import fetch_archive_from_http

doc_dir = "output_images"

fetch_archive_from_http(
    #url="https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/spirit-animals.zip",
    url='https://www.istockphoto.com/search/2/image?phrase=covid',
    output_dir=doc_dir,
)

#['Analytical','Argumentative','Definition','Compare','Cause and Effect','Interpretative','Experimental','Survey']