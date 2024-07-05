# LangChain Web Scraper and QA Bot

# Free LangChain Web Scraper and QA Bot

This project demonstrates how to build a web scraping and question-answering bot using LangChain, OpenAI's GPT model, and HuggingFace embeddings. The bot scrapes content from a specified webpage, processes the content, and then answers questions based on the scraped data.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [License](#license)

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Node.js (version 16 or later)
- npm or yarn
- A HuggingFace API Key
- An OpenRouter API Key

## Installation

1. Clone the repository:
    
    ```
    sh Copy code
    git clone https://github.com/your-username/langchain-web-scraper-qa-bot.git
    cd langchain-web-scraper-qa-bot
    
    ```
    
2. Install the dependencies:
    
    ```
    shCopy code
    npm install
    # or
    yarn install
    
    ```
    
3. Create a `.env` file in the root directory and add your HuggingFace API key:
    
    ```
    plaintextCopy code
    HUGGINGFACE_API_KEY=your_huggingface_api_key
    
    ```
    

## Usage

To run the main script:

```
shCopy code
node index.js

```

This script performs the following steps:

1. Loads the model using OpenAI's API.
2. Scrapes content from the specified webpage using Cheerio.
3. Splits the scraped content into manageable chunks.
4. Generates embeddings for the content using HuggingFace.
5. Creates a vector store to manage the embeddings.
6. Constructs a retrieval chain to answer questions based on the scraped content.
7. Outputs the response to a sample question.

## Configuration

You can customize various aspects of the script:

- **Model Configuration**: Change the `modelName` and `baseURL` in the `ChatOpenAI` instantiation.
- **Text Splitting**: Adjust `chunkSize` and `chunkOverlap` in `RecursiveCharacterTextSplitter` to handle different document structures.
- **Retriever Configuration**: Modify `k` in `vectorstore.asRetriever` to control how many chunks are retrieved for context.

### Example Configuration

In `index.js`:

```
jsCopy code
const model = new ChatOpenAI({
  modelName: "google/gemma-2-9b-it:free",
  verbose: true,
  configuration: { baseURL: "https://openrouter.ai/api/v1" },
});

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 200, // Increase chunk size for more context
  chunkOverlap: 50, // Increase overlap to ensure continuity
});

```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for more details.
