#!/bin/bash
docker build -t rag_chatbot_inference .
docker tag rag_chatbot_inference:latest axelsirota/rag_chatbot_inference:latest
docker push axelsirota/rag_chatbot_inference:latest
