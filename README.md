# MLInsightLab-Model-Container

This repository contains the source code for the **ML Insight Lab Model Container**, a lightweight service used for on-demand model deployment within the ML Insight Lab Platform.

## Overview

The **Model Container** is not accessed directly by users. Instead, it is dynamically instantiated and managed by the platform — specifically orchestrated by the **API Hub** — to serve trained machine learning models in response to deployment requests.

Each container instance:

- **Hosts a specific version of a trained ML model**
- **Exposes a standardized prediction API**
- **Is launched on-demand** based on platform deployment workflows
- **Integrates with platform-wide logging and monitoring systems**
- **Operates in an isolated, self-contained environment** for security and reproducibility

This design allows for scalable, reproducible, and low-latency model serving without user intervention in the container lifecycle.

## Build Status

[![Model Container CI](https://github.com/mlinsightlab/mlinsightlab-model-container/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/mlinsightlab/mlinsightlab-model-container/actions/workflows/docker-publish.yml)

This service is automatically built and published via GitHub Actions. You do **not** need to clone or run this repository manually unless you are contributing to the model container logic or extending its runtime behavior.

## When to Use This Repository

You should pull and work on this repository **only** when:

- You are modifying the base runtime environment for deployed models
- You are adding support for new model types or inference interfaces
- You are customizing how models are loaded, initialized, or served
- You are debugging model serving issues related to the container

All lifecycle management (startup, shutdown, scaling) is handled by the platform — not by end users.

---

We welcome contributions and suggestions via issues or pull requests, especially around extensibility and model compatibility.

