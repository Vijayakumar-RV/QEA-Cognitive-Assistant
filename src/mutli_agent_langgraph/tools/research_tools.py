from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import ArxivAPIWrapper
from newspaper import Article
from langgraph.prebuilt import ToolNode
import trafilatura
from bs4 import BeautifulSoup
import os
import streamlit as st

def tavily_web_search(query: str)->list:
    """uses Tavily to search the web and returns with url"""
    print("Tavily Search Called")
    tavily = TavilySearch(k=5)
    return tavily.run(query)

def serpAPI_search(query:str):
    """uses SerpAPI Wrapper for comparison using a web search"""
    print("SerperAPI Search Called")
    serp_api_key = os.getenv("SERPER_API_KEY")
    search = SerpAPIWrapper(serpapi_api_key=serp_api_key)
    return search.run(query)

def search_arxiv_papers(query:str):
    """uses arxiv tool and get the relevant papers"""
    print("ARXIV Called")
    arxiv = ArxivAPIWrapper()
    return arxiv.run(query + "Provide a breif Summary and then link of the Paper")

def search_wikipedia(query: str) -> str:
    """Returns a summary from Wikipedia for general queries."""
    print("Wiki Called")
    wiki = WikipediaAPIWrapper()
    return wiki.run(query)

def summarize_article_from_url(url: str) -> str:
    """Fetches and summarizes content from a news or blog article URL."""
    print("Summarize the article from URL ")
    import nltk
    nltk.download('punkt')
    try:
        article = Article(url)
        article.download()
        article.parse()
        article.nlp()
        return f"Title: {article.title}\nSummary: {article.summary}"
    except Exception as e:
        return f"Failed to summarize article: {str(e)}"
    
def extract_webpage_content(url: str) -> str:
    """Extracts clean text from a web page using trafilatura."""
    print("Extract web content")
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        return trafilatura.extract(downloaded, include_comments=False)
    else:
        return "Failed to fetch or parse URL."

def export_research_markdown(title: str, content: str) -> str:
    """Saves the research content as a markdown file locally."""
    print("Export research mark down called")
    filename = f"research_{title.replace(' ', '_').lower()}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n{content}")
    return f"Saved as {filename}"

def get_tools():
    """Return the list of tools to be used by the chatbot"""
    tools=[tavily_web_search,
           serpAPI_search,
           search_wikipedia,
           extract_webpage_content,
           export_research_markdown,
           search_arxiv_papers]
    
    return tools

def create_tool_node(tools):
    """
    creates and return tool nodes for chatbot
    """
    return ToolNode(tools=tools)

