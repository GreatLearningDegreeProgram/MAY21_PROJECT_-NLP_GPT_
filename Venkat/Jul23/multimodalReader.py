#from haystack.telemetry import tutorial_running
from haystack.document_stores import InMemoryDocumentStore
from haystack import Document
from haystack.nodes.retriever.multimodal import MultiModalRetriever
import os
from haystack import Pipeline
from config import EMBEDDING_DIM,MULTIMODAL_IMG_DIR,QRY_EMBEDDING_MODEL,QRY_TYPE,DOC_EMBEDDING_MODELS



#tutorial_running(19)

def getImage(query):
    print(query,'........................')
    document_store = InMemoryDocumentStore(embedding_dim=EMBEDDING_DIM)
    images = [Document(content=f"{MULTIMODAL_IMG_DIR}/{filename}", content_type="image") for filename in os.listdir(MULTIMODAL_IMG_DIR) ]
    document_store.write_documents(images)
    print(images)
    retriever_text_to_image = MultiModalRetriever(
    document_store=document_store,
    query_embedding_model=QRY_EMBEDDING_MODEL,
    query_type=QRY_TYPE,
    document_embedding_models={"image": DOC_EMBEDDING_MODELS},)

    document_store.update_embeddings(retriever=retriever_text_to_image)
    pipeline = Pipeline()
    pipeline.add_node(component=retriever_text_to_image, name="retriever_text_to_image", inputs=["Query"])
    results = pipeline.run(query=query, params={"retriever_text_to_image": {"top_k": 1}})
    results = sorted(results["documents"], key=lambda d: d.score, reverse=True)

    images_array = [doc.content for doc in results]
    scores = [doc.score for doc in results]

    return images_array,scores


    