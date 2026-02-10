import os
from dotenv import load_dotenv
from scraper import  fetch_website_contents
from openai import OpenAI
from openai import RateLimitError

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")

OPENAI_MODEL = None
openai_client = None

if api_key and api_key.startswith("sk-proj-") and len(api_key) > 10:
    OPENAI_MODEL = "gpt-5-nano"
    openai_client = OpenAI()
    print("Using OpenAI:", OPENAI_MODEL)

OLLAMA_MODEL = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
ollama_client = OpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key="ollama"
)
print("Using Ollama:", OLLAMA_MODEL)

class ArticleSummarizer:
    def __init__(self,Article, url):
        self.Article = Article
        self.url = url

    def fetch_page_and_all_relevant_links(self, url):
        contents = fetch_website_contents(url)
        result = f"## Landing Page:\n\n{contents}\n## Relevant Links:\n"

        return result

    summarizer_system_prompt = """
    When I provide you with a URL to a research article from any academic source 
    (such as PubMed, arXiv, Elsevier, PNAS, Nature, Science, IEEE, Springer, Wiley, BMC, MDPI, or any other scholarly database), 
    please follow these steps:

    1. **Access and read the full content** of the research article from the provided URL
    2. **Extract and analyze** all sections of the paper thoroughly
    3. **Provide a comprehensive summary** that includes:

    - **Title and Authors**: The complete title and list of authors
    - **Publication Details**: Journal name, publication date, DOI (if available)
    - **Abstract Summary**: A concise overview of the abstract in your own words
    - **Introduction/Background**: The research problem, context, and motivation for the study
    - **Research Objectives**: The specific aims, hypotheses, or research questions
    - **Methodology**: 
        - Study design and approach
        - Sample size and participants (if applicable)
        - Data collection methods
        - Analysis techniques and tools used
    - **Key Findings/Results**: 
        - Main outcomes and discoveries
        - Statistical significance (if mentioned)
        - Data patterns and trends
    - **Discussion**: Interpretation of results and their implications
    - **Limitations**: Any constraints or weaknesses acknowledged by the authors
    - **Conclusions**: Final takeaways and the authors' main conclusions
    - **Future Directions**: Suggestions for future research (if mentioned)
    - **Significance**: Why this research matters to the field

    4. **Structure your summary** in a clear, organized format with appropriate headings and subheadings
    5. **Maintain academic accuracy** while making the content accessible and easy to understand
    6. If you cannot access the full text, clearly state what portions you were able to read and summarize accordingly

    Please provide the summary in a structured format that allows me to quickly grasp the essence of the research while also having
    access to detailed information about each component of the study.
    """

    def get_user_prompt(self):
        user_prompt = f"""
            You are looking at a Article called: {self.Article}
            Here are the contents page of given article 
            use this information to build a summary of the article in markdown without code blocks.\n\n
            """
        user_prompt += self.fetch_page_and_all_relevant_links(self.url)
        user_prompt = user_prompt[:10_000] # Truncate if more than 5,000 characters
        return user_prompt
        

    def stream_article(self):
        try:
            stream = openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": self.summarizer_system_prompt},
                    {"role": "user", "content": self.get_user_prompt()}
                ],
                stream=True
            )

        except RateLimitError:
            # OpenAI quota exhausted → fallback
            stream = ollama_client.chat.completions.create(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": self.summarizer_system_prompt},
                    {"role": "user", "content": self.get_user_prompt()}
                ],
                stream=True
            )

        output = ""
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            output += token

        return output
            
            
if __name__ == "__main__":
    summarizer = ArticleSummarizer(
        Article="A Comprehensive Review on Machine Learning Techniques for Cybersecurity",
        url="https://arxiv.org/abs/2103.00020"
    )
    summarizer.stream_article()
