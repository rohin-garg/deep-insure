# Health Insurance Plan Analyzer -- HackMIT '25

Health insurance plans are notoriously difficult, with many Americans dissatisfied with how opaque and confusing they are. Our project makes these plans understandable by turning 1000+ page PDFs into clear, searchable, and interactive summaries.

## Inspiration
Health insurance documents are dense, legalistic, and often inaccessible to the average reader. We set out to create an agent that:
- Summarizes insurance plans into plain, easy-to-read language  
- Provides AI-powered search with citations to the original text  
- Offers qualitative feedback based on real user experiences  

## What It Does
The app takes in a URL to an insurance plan and transforms it into a wiki-like interface:
- Summarizes complex terms in simple language  
- Highlights key policy points without jargon  
- AI-powered chatbot answers questions with citations  
- Search across both policy text and relevant online resources  

## How We Built It
Building a research tool for long insurance plans required several technical approaches:
- **Context Engine**: Hybrid classical and embedding-based retrieval for cheap, accurate pruning of irrelevant text  
- **Model Fine-Tuning**: Synthetic data generation and fine-tuning of Gemma-2B for lightweight relevance classification  
- **Frontend**: React, Vite, and Tailwind for a responsive user experience  

## Individual Contributions
- Brian: Fine-tuned Gemma-2B classifier to detect red flags in policies  
- Rohin: Backend API, context engine, LLM integration and optimization  
- Jeremy: Frontend-backend integration, general UX improvements  
- William: Frontend UI design, interactivity, and overall polish  

## Challenges
- Classifier training required tuning hyperparameters, particularly a much higher learning rate  
- Deployment on Modal was slow until the algorithm was restructured to prevent repeated model weight loading  
- Handling extremely long documents while keeping relevant information accessible required hybrid algorithms  

## Accomplishments
- Functional, polished frontend and user experience  
- Successful fine-tuning of Gemma-2B to insurance-specific tasks  
- Major reductions in inference time and API cost through algorithmic optimization and parallelism  

## What We Learned
- The difficulty of integrating multiple APIs into a cohesive system  
- The importance of clean user experiences in technical tools  
- Techniques for optimizing models to process massive documents efficiently  
- Managing complexity is both a technical and design problem  

## What's Next
- Scaling performance with better compute resources  
- Extending the system to other types of dense, confusing documents:
  - Property insurance  
  - Terms and conditions  
  - Legal contracts and claims  
- Building towards a paralegal-like tool to assist claimants navigating complex policies  

## Tech Stack
- Backend: Python, FastAPI, LLMs (Gemma-2B fine-tuned)  
- Frontend: React, Vite, TailwindCSS  
- Infrastructure: Modal for model deployment and scaling  
- Search/Context Engine: Hybrid classical and embedding-based retrieval  
