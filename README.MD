---
runme:
  id: 01HW4HCKHZ0WJJH10A4Q6HSZ7G
  version: v3
---

```markdown {"id":"01HW4HCKHY8DRZKJ8Y3RYCHJAE"}
# Project Codebase

This project implements a web crawling and information retrieval system using Scrapy for web crawling, Scikit-Learn for document indexing with TF-IDF, and Flask for query processing.

## Overview

This project combines web crawling, document indexing, and query processing to retrieve and search web content efficiently. The system uses Scrapy to crawl web documents, Scikit-Learn to create an inverted index using TF-IDF representation, and Flask to provide a web interface for querying the indexed documents.

## Components

- Scrapy Crawler: Initializes a web crawler to download web documents in HTML format.
- Scikit-Learn Indexer: Constructs an inverted index using TF-IDF representation for efficient search indexing.
- Flask Processor: Implements a Flask-based processor for handling free text queries and retrieving top-ranked search results.

## Requirements

- Python 3.10+
- Scrapy 2.5+
- Scikit-Learn 1.2+
- Flask 2.0+

## Setup and Usage

1. Install Dependencies:
   pip install scrapy scikit-learn Flask

```

2. Crawling Web Content:

- Initialize the Scrapy crawler (`my_crawler.py`) with seed URLs, max pages, and depth settings:

```bash {"id":"01HW4HCKHZ0WJJH10A4EZZCXGR"}
scrapy runspider my_crawler.py

```

3. Indexing Documents:

- Run the indexer (`indexer.py`) to create an inverted index using TF-IDF representation:

```bash {"id":"01HW4HCKHZ0WJJH10A4HC98R9P"}
python indexer.py

```

4. Starting Flask Server:

- Launch the Flask server (`processor.py`) to handle queries and provide search results:

```bash {"id":"01HW4HCKHZ0WJJH10A4MTNFED4"}
python processor.py

```

5. Interacting with the API:

   - Use HTTP POST requests to query the Flask server (`http://127.0.0.1:5000/search`) with JSON payloads containing search queries.

# Query

To perform a sample query, send a POST request with JSON data:

```bash {"id":"01HW4HCKHZ0WJJH10A4NMGJDRC"}
curl -X POST -H "Content-Type: application/json" -d '{"query": "Lane isn't content to simply wait"}' http://127.0.0.1:5000/search

```

## Project Structure

- `my_crawler.py`: Defines the web crawler using Scrapy to download web documents.
- `indexer.py`: Computes TF-IDF scores and builds an inverted index for the crawled documents.
- `processor.py`: Implements a Flask-based processor to handle queries and retrieve search results.
