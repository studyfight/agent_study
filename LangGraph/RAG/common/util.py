from langchain_core.documents import Document

def print_red(s: str):
    print('\033[91m{}\033[0m'.format(s))

def print_docs(docs: list[Document]):
    for i, doc in enumerate(docs, start=1):
        print(f"---- 第{i}段 ----")
        print(doc.page_content)

def print_red_docs(docs: list[Document]):
    for i, doc in enumerate(docs, start=1):
        print_red(f"---- 第{i}段 ----")
        print_red(doc.page_content)