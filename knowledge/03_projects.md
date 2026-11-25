# Projects

## Ask Waheed: portfolio chatbot (RAG)
The chatbot on Waheed's portfolio website, which answers questions about his experience and projects.
- Retrieval-augmented generation over a markdown knowledge base, using hybrid search: BM25 keyword search plus dense embeddings (BGE-small), merged with reciprocal rank fusion.
- FastAPI backend deployed as a serverless function on Vercel (also runs in Docker); LLM calls through an OpenAI-compatible API (Groq, with Gemini as fallback).
- Guardrails: answers only from retrieved context, cites its sources, refuses off-topic requests and resists prompt injection; per-visitor rate limiting.
- Includes an evaluation script that checks retrieval hit rate, answer facts and refusal behaviour.

## Solar Production and Load Prediction
August 2024. Final project of the GIKI DNN Bootcamp, which won 2nd place among 53 participants.
- Time-series forecasting of solar power generation and electricity load.
- Integrated real-time weather data and engineered features such as sunrise/sunset and seasonal indicators.
- Reduced mean absolute error (MAE) by 11% through feature engineering and one-hot encoding.

## AI-Powered Age and Gender Detection (final year project)
June 2023 to June 2024. BS final year project at Virtual University of Pakistan.
- Detects faces and predicts age group and gender from images, built with OpenCV, scikit-learn, Python and Flask.
- Reached 92% test accuracy on diverse image datasets.

## Dental X-Ray Segmentation (in progress)
Independent research project, started June 2025, now being rebuilt from scratch for publication and a public demo.
- Segments individual teeth in panoramic dental X-rays.
- Evaluated YOLOv8 segmentation, Mask R-CNN and watershed post-processing; analysed failure cases such as ambiguous tooth boundaries.
- Studying transformer-based segmentation (Mask2Former-style models) to overcome CNN limitations.
- Long-term interest: medical computer vision.
