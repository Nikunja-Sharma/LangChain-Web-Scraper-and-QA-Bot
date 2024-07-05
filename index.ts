import { ChatOpenAI } from "@langchain/openai";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { createRetrievalChain } from "langchain/chains/retrieval";
import dotenv from "dotenv";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
dotenv.config();

async function main() {
  // Instantiate Model
  const model = new ChatOpenAI({
    modelName: "google/gemma-2-9b-it:free",
    verbose: true,
    configuration: { baseURL: "https://openrouter.ai/api/v1" },
  });

  // Create prompt
  const prompt = ChatPromptTemplate.fromTemplate(
    `Answer the user's question from the following context: 
    {context}
    Question: {input}`
  );

  // Use Cheerio to scrape content from webpage and create documents
  const loader = new CheerioWebBaseLoader("https://ai.nikunja.online/about");
  const docs = await loader.load();
  console.log("Loaded Documents:", docs);

  // Text Splitter
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 200, // Increase chunk size for more context
    chunkOverlap: 50, // Increase overlap to ensure continuity
  });
  const splitDocs = await splitter.splitDocuments(docs);
  console.log("Split Documents:", splitDocs);

  // Ensure there's meaningful content in split documents
  if (splitDocs.length === 0 || splitDocs.every(doc => doc.pageContent.trim() === "")) {
    console.error("No meaningful content extracted from the webpage.");
    return;
  }

  // Instantiate Embeddings function
  const embeddings = new HuggingFaceInferenceEmbeddings({
    apiKey: process.env.HUGGINGFACE_API_KEY,
    model: "sentence-transformers/all-MiniLM-L6-v2",
  });

  // Create Vector Store
  const vectorstore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);
  console.log("Vector Store Created");

  // Create a retriever from vector store
  const retriever = vectorstore.asRetriever({ k: 5 }); // Increase 'k' for more context
  console.log("Retriever Created");

  // Create Chain
  const chain = await createStuffDocumentsChain({
    llm: model,
    prompt,
  });
  console.log("Chain Created");

  // Create a retrieval chain
  const retrievalChain = await createRetrievalChain({
    combineDocsChain: chain,
    retriever,
  });
  console.log("Retrieval Chain Created");

  // Invoke Chain
  const response = await retrievalChain.invoke({
    input: "What is the page about? Who is it?",
  });
  console.log("Response:", response.answer);
}

main().catch(console.error);
