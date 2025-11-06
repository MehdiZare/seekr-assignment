## CI/CD, Infrastructure, and Operational Resilience

**Automated Delivery Pipeline (GitHub Actions)**

- **Build & Validation:** On push or PR, the backend Docker image is built, tested for summarization fidelity, fact-check accuracy, and LangGraph state recovery, then pushed with immutable tags to **Amazon ECR**. The Next.js frontend is compiled via `next build` and deployed through **Pulumi-managed Vercel resources**.

- **Isolated Preview Environments:** Each PR triggers **Pulumi** to provision a complete stack: Fargate task (0.25 vCPU, 0.5 GB), ElastiCache Redis (cache.t4g.micro), Supabase database branch, and S3 prefix. The backend service URL is injected as an environment variable (`NEXT_PUBLIC_API_URL`) into the Vercel deployment, linking the frontend directly to its matching preview backend. Cloudflare routes `pr-<id>.domain.com` for full-stack testing.

- **Production Promotion:** Merges to `main` instantly promote changes via Pulumi. For internal use, deployments are atomic—no staged rollouts. Rollbacks use Pulumi stack restore or ECR image reversion.

- **Resource Lifecycle:** Preview environments auto-destroy on PR closure via GitHub webhooks. ECR retains only the latest 3 images per environment.

---

**Infrastructure Components & Cost Profile**  
*(~500 episodes/month, internal usage)*

| Component | Service | Configuration | Monthly Cost |
|---------|--------|---------------|--------------|
| **Compute** | AWS Fargate | 0.25 vCPU, 0.5 GB × 200 hrs | ~$12 |
| **State Store** | ElastiCache Redis | cache.t4g.micro (single-AZ default) | ~$18 |
| **Object Storage** | Amazon S3 | 50 GB outputs + audit trails | ~$1.15 |
| **Database** | Supabase (Pro) | 1 GB, 5 branches | $25 |
| **Frontend** | Vercel Pro | 100 GB bandwidth | $20 |
| **DNS & Security** | Cloudflare Pro | WAF, DDoS, caching | $20 |
| **IaC & Registry** | Pulumi Team + ECR | 100 runs, 10 GB storage | $29 + $1 |
| **Data Transfer** | AWS + Vercel | 50 GB egress | ~$4.50 |
| **LLM Inference** | Meta Maverick (primary) | ~2.5M tokens @ 3K RPM | $0 |
| | OpenAI/Anthropic (fallback) | On-demand | Pay-per-use |
| | Groq (optional) | Low-latency paths | Pay-per-token |

**Total Base Cost (excl. fallback LLM): ~$130–$150**

---

**Fault Tolerance & Recovery Strategy**

- **Compute & Cache:** Fargate and ElastiCache default to single-AZ for cost efficiency. Pulumi parameters allow one-line multi-AZ activation when needed.

- **Database Resilience:** Supabase Pro includes multi-AZ replication. Daily backups are exported to S3 Standard-IA (~$0.0125/GB-month) for durable, low-cost retention. Cross-region replication is configurable via Pulumi.

- **Storage & Audit:** S3 enables versioning and Object Lock for immutable outputs and audit logs. 11 9s durability ensures data integrity.

- **Model Availability:** Meta Maverick (free, 3000 RPM) serves primary inference. Automatic fallback to OpenAI, Anthropic, or Groq activates on rate limits or timeouts via agent routing.

- **Observability:** Logs flow to CloudWatch (7-day retention, ~$3/month). Fargate health checks trigger task replacement. Alarms monitor CPU > 70%, Redis memory pressure, and LLM latency.

Preview environments deliver tightly coupled frontends and backends with dynamic URLs, enabling fast, reliable internal validation. Resilience scales on demand via infrastructure-as-code.