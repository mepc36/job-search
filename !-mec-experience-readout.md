# Martin Connor — Engineering Work Summary

**Prepared for:** Software Engineer, ML Platform — Luma AI

This document summarizes work Martin Connor personally built, verified by reading his git commits and source code across multiple repositories. It is intended as a factual basis for resume generation. Each section opens with platform context, followed by what Martin specifically built.

## RapBot.ai (Jan 2019 – May 2026)

RapBot.ai was a production generative AI music platform Martin founded and operated end-to-end. The platform allowed users to generate original rap songs — custom lyrics, cloned vocals, and AI-composed backing music — by combining a suite of custom-trained and fine-tuned deep learning models running on GPU infrastructure. Martin owned everything: the user-facing web and mobile product, all backend APIs, the GPU inference infrastructure, CI/CD pipelines, and every ML training pipeline. This was a real production system serving paying users, not a research prototype.

### GPU Inference Scheduling System

**Platform context:** RapBot's core inference pipeline required coordinating multiple expensive GPU model invocations per user request — lyric generation, TTS synthesis, music generation, and audio processing — each routed to different GPU workers that could be on different providers depending on cost and availability.

**What Martin built:** A PostgreSQL-backed job scheduling system to manage all GPU inference workloads across the platform. Each job was tracked through its full lifecycle — queued, dispatched, processing, completed, failed — with configurable retry limits, maximum retry counts, and expiry windows defined per job type. Workers were designed to self-reschedule: after finishing one job they would immediately claim the next available one, keeping GPU utilization high.

When a job completed, the scheduler fired real-time push events to the user's browser via Pusher — a managed WebSocket transport layer. The completion hook in the job queue called the Pusher server SDK to trigger a session-scoped channel event keyed to the user's session ID, pushing a job-complete payload directly to the subscribed client and eliminating polling entirely. In development, channels were isolated per session to support concurrent users; in production a shared channel was used. Errors were also pushed via Pusher (with an error flag) while simultaneously posting a structured error payload to a Slack webhook for operational alerting. The Pusher client was implemented as a server-side singleton to avoid re-instantiation on every job event.

The scheduler also served as the dispatch layer for routing workloads across multiple GPU providers, making the system provider-agnostic by design. This maps directly to Luma's requirement for "robust and sophisticated scheduling systems to manage jobs based on cluster availability and user priority, ensuring optimal leverage of expensive GPU resources."

### Multi-Provider GPU Fleet Management and Inference Cost Optimization

**Platform context:** Serverless GPU providers vary significantly in cost, latency, and availability. RapBot ran inference on multiple providers simultaneously and needed to shift workloads between them without re-architecting the application.

**What Martin built:** A provider-agnostic inference dispatch layer that abstracted away the specifics of any individual GPU provider. Martin migrated RapBot's production inference workloads from one serverless GPU provider to another — achieving a 9% reduction in GPU compute costs — without disrupting production. The dispatch layer used async polling to track job status across providers and handled artifact handoff via blob storage: GPU workers wrote output files to blob storage, and the scheduler retrieved them regardless of which provider processed the job. This is the same fleet management and multi-provider orchestration pattern Luma describes for "managing and optimizing complex inference workloads at scale, operating across multiple clusters and hardware providers."

### Model Hotswapping and CI/CD Pipeline

**Platform context:** RapBot deployed multiple AI models to production simultaneously — TTS, lyric generation, music generation, audio alignment — each with its own versioning, dependencies, and Docker image. Keeping all of these current without downtime required a disciplined deployment pipeline.

**What Martin built:** A fully automated CI/CD pipeline that, on every push to the main branch, injected secrets from an encrypted store, synchronized the updated codebase to the production server over SSH (explicitly excluding build artifacts, media files, and native mobile build directories to keep transfers fast), then remotely rebuilt Docker images and restarted containerized services — achieving zero-downtime model updates without any manual intervention. Model checkpoints and inference artifacts were stored in blob storage and transferred to GPU instances for cold-start provisioning. This is equivalent to the "resilient artifact store and CI/CD pipeline" and "hotswapping models on GPU workers" responsibilities in Luma's JD, applied to a real multi-model production fleet.

### Containerized Production Inference Stack

**Platform context:** RapBot's production inference services needed to be portable across providers, reproducible across environments, and isolated from each other to prevent dependency conflicts between models with different requirements.

**What Martin built:** A multi-service production deployment using Docker Compose, with each AI model and supporting service running in its own container. A Traefik reverse proxy sat in front of all services handling TLS termination and routing. This containerization strategy directly enabled the multi-provider fleet management described above — Docker images could be built once and deployed to any provider that supported container execution.

### FFmpeg Multimedia Processing Pipelines

**Platform context:** Every generated RapBot song required non-trivial audio and video processing — combining dozens of individually synthesized audio segments into a synchronized output, trimming audio for model training data, and handling user-uploaded audio on the client side. FFmpeg was the backbone of all three.

**What Martin built:** Three distinct FFmpeg pipelines serving different parts of the system:

- **Server-side audio/video compositing pipeline:** A complex FFmpeg filter graph that ingested individually synthesized per-word audio stems — one clip per word of the generated rap lyric — and composited them into a single synchronized audio track. The graph applied per-channel stereo panning, decibel-level amplitude control, and millisecond-accurate delay offsets per stem to achieve proper timing and spatial positioning. The resulting audio was combined with an animated lyric overlay to render a final MP4 video, which was uploaded to blob storage for delivery.
- **Docker-containerized static binary for training data preparation:** A statically compiled FFmpeg binary baked directly into a Docker image used in the forced-alignment service. This binary performed sample-accurate audio trimming to extract individual vocal samples from longer recordings — a key step in building the voice cloning training dataset.
- **Client-side FFmpeg WebAssembly:** FFmpeg compiled to WASM and running entirely in the user's browser, used to transcode microphone recordings before upload. This eliminated a server round-trip for audio preprocessing and reduced backend load.

### PyTorch Model Training: RADTTS Voice Cloning + GPU Provisioning

**Platform context:** RapBot's voice synthesis feature required cloning a specific rapper's voice from a limited set of audio samples. The underlying model was NVIDIA's RADTTS — a normalizing-flow neural TTS architecture that maps text and speaker embeddings to mel spectrograms. Martin forked the RADTTS repository, built the training dataset pipeline, ran the full fine-tuning process, and wrote the end-to-end GPU provisioning runbook for deploying inference on AWS.

**What Martin built:** A complete training and deployment pipeline on top of RADTTS using PyTorch. He first built a dataset curation pipeline: Meta's Demucs source separation model isolated vocal tracks from full songs, a forced-alignment tool identified word-level boundaries, and FFmpeg cut those boundaries into individual samples stored in blob storage. This produced a curated set of 465 vocal samples.

For training, Martin configured warm-start transfer learning starting from a pretrained RADTTS checkpoint, selectively froze early layers to preserve low-level acoustic features while fine-tuning higher-level speaker characteristics, and used PyTorch's automatic mixed-precision training with a GradScaler to maintain numerical stability during half-precision training. He used the RAdam optimizer with gradient clipping and ran training across multiple GPUs using PyTorch DistributedDataParallel. The fine-tuned model produced voice clone output that was demoed live to investors.

Martin also wrote and maintained the full GPU provisioning runbook for deploying RADTTS inference on AWS. The procedure covered: launching a g4dn.xlarge Ubuntu 22.04 instance with 104 GB of storage, locking PEM file permissions, SSHing into the instance, installing the CUDA toolkit configured for the specific Linux target via the NVIDIA installer, verifying GPU availability via PyTorch and nvidia-smi, transferring fine-tuned model checkpoints and training data to the instance via SCP, and installing all RADTTS Python dependencies. This represents end-to-end hands-on GPU instance provisioning, not just model training — directly relevant to Luma's requirement for deep familiarity with managing GPU resources on cloud hardware.

### WER-Based Voice Clone Evaluation

**Platform context:** Evaluating whether the RADTTS voice clone was improving or degrading across training iterations required an objective, repeatable measure — with one engineer, no labeling budget, and no existing benchmark for cloned rap vocals.

**What Martin built:** A round-trip WER evaluation pipeline: feed the fine-tuned model a known transcript, synthesize audio, run that audio through speech-to-text, and compute word error rate between the STT output and the original transcript. The known transcript made ground truth free, requiring no human labeling. Training runs were tracked in Weights & Biases, so WER deltas mapped to specific checkpoint and data changes, and the resulting quality signal decided which checkpoints shipped.

### LLM Fine-Tuning: GPT-J Lyric Generation

**Platform context:** RapBot's lyric generation feature needed to produce rap lyrics that rhymed correctly, matched a given topic, and felt stylistically authentic. Martin first used OpenAI's GPT-3 Davinci via API fine-tuning, then replaced it with a self-hosted open-source model to eliminate per-token API costs and gain full control over inference.

**What Martin built:** Martin first fine-tuned GPT-3 Davinci on a custom rap couplet corpus, formatted as completion-only pairs and filtered through a rhyme density scoring function to retain only high-quality examples. He then replaced this entirely with a self-hosted GPT-J 6B model, quantized to INT8 using bitsandbytes to reduce GPU memory requirements, and running on his own inference infrastructure. For the GPT-J fine-tune, he redesigned the training data schema into 8 parallel structured task formats — each using custom delimiter tokens to teach the model to perform multiple related subtasks (topic-to-rhyme, rhyme-to-lyric-line, grapheme-to-phoneme mapping) from a single fine-tuning run. At inference time he added a three-pass constrained beam search pipeline with hard filtering on the final word of each line to enforce rhyme. The result was a 64% reduction in failed lyric generation attempts.

### MusicGen Fine-Tuning

**Platform context:** RapBot needed to generate hip-hop backing tracks that felt stylistically consistent with a target artist's catalog. Martin fine-tuned Meta's MusicGen model on custom datasets rather than using it off-the-shelf.

**What Martin built:** Martin fine-tuned MusicGen on two curated Dr. Dre hip-hop datasets (30 and 78 songs respectively) using a cloud training API. He built a log-parsing pipeline that extracted and deduplicated musical descriptor labels from raw training output — BPM, genre tags, instrument labels, and mood descriptors — and organized them into per-label files. He then built a data-driven prompt sampler that randomly assembled inference prompts from those learned labels using defined ranges (1 BPM value, 1–3 genre tags, 1–4 instruments, 1–5 mood descriptors), producing varied but genre-consistent generations at inference time.

### 5-Service Containerized Data Pipeline (Training Corpus Construction)

**Platform context:** Building a high-quality TTS voice clone required a large, clean, word-level-aligned dataset of rap vocals. No such dataset existed publicly, so Martin built the full data pipeline to construct one from scratch — a 5-service containerized system that scraped, downloaded, cleaned, queued, and aligned approximately 400,000 songs.

**What Martin built:**

- **Lyrics scraper:** Scraped song lyrics by artist, album, and song title, with content hashing to deduplicate and avoid reprocessing known records.
- **Audio fetcher:** Searched YouTube for each song and selected the highest-quality audio source using quality-based filtering, then downloaded and stored the audio.
- **Transcript cleaner:** Normalized unicode characters, stripped structural markup (verse/chorus labels), and stored clean transcripts for alignment.
- **Job prioritizer:** A PostgreSQL-backed scheduler that filtered work by artist, duration, and processing status to queue songs for the alignment step.
- **Aligner:** Ran Demucs source separation to isolate the vocal track from downloaded audio, then ran forced alignment to produce word-level timestamps mapping each lyric word to its position in the audio.

All five services were Docker-containerized and coordinated by the job scheduling system described above. The output was a corpus of approximately 400,000 aligned songs used as training data for TTS and lyric generation models.

## NBCUniversal (Aug 2024 – Present)

NBCUniversal built an internal enterprise AI agent platform called Prism. It is a full-stack web application where NBCU employees interact with AI agents configured for specific business workflows — each agent has a system prompt, a selected LLM, connectors to enterprise data sources like SharePoint, and a set of callable tools. The backend is FastAPI (Python), the frontend is Next.js (TypeScript), and the system runs on Azure: Azure Container Apps for compute, Azure CosmosDB for conversation history, Azure Key Vault for secrets management, and Azure OpenAI as the LLM provider. The platform serves agents for multiple NBCU business units including legal, HR, ad sales, and media operations.

Martin Connor's verified contributions span hundreds of commits across the full codebase — backend inference, retrieval, tool calling, frontend analytics, document ingestion, and access management.

### LLM Inference Layer Migration (Live in Production)

**Platform context:** Prism's entire chat experience — streaming responses, tool calls, file handling, multi-turn conversation history — runs through a central inference layer that communicates with Azure OpenAI. This layer handles the full complexity of LLM inference: message formatting, content type normalization, streaming event parsing, error recovery, and tool call orchestration.

**What Martin built:** A large-scale migration of the inference layer from OpenAI's Chat Completions API to the newer Responses API — a roughly 4,900-line change touching the core inference module and surrounding services. This was not a simple API swap: the Responses API uses a fundamentally different streaming event model. Martin handled the full taxonomy of streaming events — response creation, output item addition, function call argument streaming, completion, failure, and error conditions — building the state machine needed to translate these into coherent streaming output for the frontend. He also built a message transformation layer that takes multi-turn conversation history stored in CosmosDB and reformats it for the Responses API, including content type remapping and system prompt injection. He wrote 2,500+ lines of unit tests covering both streaming and non-streaming code paths. This migration is running live in production for all NBCU employees. Martin subsequently extended the same layer to support GPT-5 when it became available.

### Hybrid Retrieval System (RAG)

**Platform context:** Prism's agents answer questions by retrieving relevant context from enterprise knowledge bases — SharePoint documents, HR policy files, legal reference data — before generating a response. The platform needed a retrieval system that could handle both conceptual semantic queries and precise keyword lookups against the same corpus.

**What Martin built:** A hybrid retrieval service combining BM25 Okapi keyword search for exact and lexical matching with dense vector cosine similarity search for semantic relevance. The two scores are combined with a tunable blend weight, allowing the system to be adjusted toward keyword precision or semantic recall depending on the agent's use case. Martin identified that the embedding model powering semantic search was being instantiated fresh on every inference request — a significant latency and resource inefficiency — and refactored it to a process-level singleton that loads once at server startup and is reused for all subsequent requests. The hybrid approach improved retrieval accuracy by 13% as measured by automated LLM-as-judge evaluations using the RAGAS framework, which Martin also set up and ran.

### LLM-as-Judge Evaluation Pipeline (Release Gating)

**Platform context:** Every Prism release required manual review of agent behavior across the platform's production agents — slow, inconsistent, and unrepeatable, with regressions able to reach employees before anyone noticed.

**What Martin built:** An LLM-as-judge evaluation pipeline using DeepEval covering 4 production agents, with per-agent test suites encoding expected behaviors judged automatically against defined criteria. Martin parallelized test execution, turning a serial afternoon of checks into minutes, and wired the results into the release process as a repeatable gate rather than an optional report — cutting manual per-release QA review time by 4 hours.

### Tool-Calling and API Architecture for AI Agents

**Platform context:** Prism agents need to call external tools and data sources during inference — fetching live data from BigQuery, querying SQL databases, calling downstream Microsoft Copilot agents, and performing document retrieval. Each tool needs typed input validation, streaming-compatible output, and clean error handling.

**What Martin built:** A decorator-based typed tool registry where each LLM-callable tool is a self-contained class with Pydantic-validated input arguments and a generator-based execution method that can yield streaming status updates to the client before returning a final result. This pattern made it easy for the team to add new tools without touching the core inference layer. Martin implemented multiple production tool handlers: hybrid search, downstream Copilot agent dispatch (authenticating against Key Vault, calling Ad Sales and HR Thrive agents), BigQuery schema retrieval, SQL aggregation via parameterized templates, and a vector search refactor into the typed registry pattern.

### Multi-Agent Platform Architecture

**Platform context:** Prism hosts multiple specialized AI agents serving different NBCU business units, each with different knowledge bases, personas, tool sets, and access control requirements.

**What Martin built:** Martin architected and deployed several production agents from scratch:

- **Atlas Assist (legal agent):** Legal research agent serving NBCU employees with questions about contracts, firm contacts, and legal resources. Martin built Legal Lookup and Firm Finder subagents that route specific query types to SharePoint-sourced knowledge bases via structured tool calls. He wrote and iterated the system prompt, added intelligent nickname handling so the agent could match formal legal names when users provided informal ones, and added prompt injection resistance.
- **HR Thrive:** HR knowledge agent built from initial configuration through production — system prompt design, RAG corpus setup, user access provisioning, and suggested prompt curation. Reduced HR Tier-1 response time 10%.
- **OmniAgent:** General-purpose agent; Martin managed access provisioning and brand identity.
- **Computer-use agent:** Replaced a brittle legacy media-fulfillment integration by autonomously completing intake workflows, cutting average response time 1.5 days.

All agents were maintained and iterated across GPT-3.5, GPT-4, and GPT-5 model generations.

### Inference Observability and Analytics

**What Martin built:** Martin instrumented the full inference request lifecycle across backend and frontend. On the backend: per-user, per-bot Datadog APM span tagging on every inference request, enabling production tracing at the individual user and agent level; distributed tracing on BigQuery context fetching and serverless document ingestion. On the frontend: Amplitude initialization with a double-init guard that waits for authenticated user session before firing, with session enrichment plugins attaching SPA session ID, chat session ID, and SSO identity to every analytics event; Datadog RUM custom actions on the message send handler; a feedback hook that walks conversation history to identify the exact message that prompted a rated response and fires correlated events to both Datadog and Amplitude; Datadog session replay capture for all production sessions.

### Document Ingestion and Knowledge Base Pipeline

**What Martin built:** Martin owned the Azure Logic Apps workflow layer orchestrating SharePoint document ingestion across multiple agent knowledge bases. Key contributions: a document exclusion flag system so documents tagged for exclusion in SharePoint are skipped during ingestion; blob cleanup logic so deleted SharePoint documents have their vectorized data removed from storage; a reference data pipeline that copies XLSX files from SharePoint, converts them to CSV via an Azure Function, and triggers vectorization. He maintained the CI/CD pipeline deploying workflow definitions across development, staging, and production environments.

### User Access Management Infrastructure

**What Martin built:** A suite of Python CLI scripts for managing user access to the platform — user upsert, access validation, deletion, bulk removal, and per-agent access grants and revocations. Used by the platform's operations team to onboard users and manage access at scale, and a prerequisite for safely rolling out new agents to specific business units before broader release.

## Hy-Vee, Inc. (Jan 2023 – Aug 2024)

Hy-Vee is a large grocery chain; Martin was a Senior Software Engineer owning core systems on its loyalty platform serving 100K+ members, delivering personalized promotions across digital and in-store points of sale.

**What Martin built:** Standardized the sign-up process across multiple web apps built with Next.js, TypeScript, and GraphQL, reducing sign-up funnel abandonment 7%. Built gamified "trip challenge" shopping experiences using MSSQL and GraphQL, growing departmental revenue 9%. Tuned high-volume promotion workflows using targeted data fetching, pagination, and Redis caching to keep offer flows resilient under high traffic.

## Waterfield Technologies (Oct 2021 – Dec 2022)

Waterfield built cloud contact-center platforms for external clients, productizing Twilio APIs into a reusable SaaS offering. Martin was a Senior Software Engineer who also led a team of 4 developers (2 senior, 2 junior) responsible for over $1.1M in department revenue — running stand-ups, performance reviews, and one-on-ones.

**What Martin built:** Defined and validated technical acceptance criteria — latency targets, throughput limits, and error budgets — via load and failure testing, proving $1.2M in contractual SLA commitments. Consolidated four separate agent workflows into a single state-driven application, cutting average call time 7 minutes. Migrated a client's call center from Avaya to a custom Twilio platform designed for stateless call flows and horizontal scaling, increasing concurrent call capacity 2.5x. Architected pre-sales solutions used as the definition of done in scopes of work.

## DrayNow, Inc. (Sep 2019 – Oct 2021)

DrayNow was a two-sided logistics marketplace connecting truck drivers with intermodal freight brokers. Martin was a Software Engineer II owning revenue-critical workflows across pricing, dispatch, invoicing, and the customer-facing mobile app.

**What Martin built:** Deployed the marketplace on Docker with AWS ECS/Fargate for horizontal autoscaling, with Traefik automating SSL certificate renewal for production URLs. Built event-driven flows on AWS SQS/SNS and Lambda microservices, including automated removal of drivers with expired licenses via the DMV's API. Replaced a bespoke invoice-processing pipeline (Lambda/S3/cron) with a Boomi integration importing invoices into NetSuite with proper retry logic and exponential backoff — cutting accounting costs $60K/year and reducing erroring documents 20%. Built versioned pricing models on S3/Python recommending guaranteed-sale prices to vendors.

### DrayNow — Guaranteed Price Model Maintenance

DrayNow's marketplace listed freight at a "guaranteed price," a model-suggested price at which a load would be picked up by a driver 98% of the time, trained on historical pricing data. Martin maintained this model in production. He monitored a rolling one-month pickup rate against the 98% target, and when the rate degraded he diagnosed and fixed it: retraining on newer data, adjusting for seasonal price shifts (e.g. holiday-season spikes), or rolling back to a prior model version stored in S3 when anomalies like mass order cancellations had polluted recent data.