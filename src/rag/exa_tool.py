import os
from typing import Optional
from langchain_exa import ExaSearchRetriever
from dotenv import load_dotenv

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(BASE_DIR, ".env"))

class ExaSearchTool:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("EXA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "EXA_API_KEY is not set. Please set the environment variable or pass api_key explicitly."
            )

    def search(
        self,
        query: str,
        num_results: int = 3,
        domain_bias: bool = True,
    ) -> list[dict]:
        """
        Performs semantic web search using langchain_exa's ExaSearchRetriever.
        Biases toward Indian government domains by default, with automatic fallback
        to open-domain search if no results are found.
        """
        gov_domains = [
            "gov.in",
            "nic.in",
            "pib.gov.in",
            "indiabudget.gov.in",
        ] if domain_bias else None

        try:
            retriever = ExaSearchRetriever(
                exa_api_key=self.api_key,
                k=num_results,
                type="auto",
                include_domains=gov_domains,
                highlights=True,
                text_contents_options={"max_characters": 1200},
            )

            docs = retriever.invoke(query)

            # If strict domain filtering yielded 0 documents, retry globally
            if domain_bias and not docs:
                return self.search(query, num_results=num_results, domain_bias=False)

            results = []
            for doc in docs:
                results.append({
                    "title": doc.metadata.get("title") or "Web Source",
                    "url": doc.metadata.get("url", ""),
                    "text": doc.page_content.strip(),
                })
            return results

        except Exception as exc:
            # Fallback to unrestricted search on filter error
            if domain_bias:
                return self.search(query, num_results=num_results, domain_bias=False)
            print(f"[!] Exa Search API error: {exc}")
            return []


if __name__ == "__main__":
    # Quick standalone sanity check
    tool = ExaSearchTool()
    test_hits = tool.search("PM Surya Ghar Muft Bijli Yojana rooftop solar scheme portal", num_results=2)
    print(f"[*] Retrieved {len(test_hits)} results from Exa via langchain_exa.")
    for hit in test_hits:
        print(f"\nTitle: {hit['title']}")
        print(f"URL  : {hit['url']}")
        print(f"Text : {hit['text'][:150]}...")