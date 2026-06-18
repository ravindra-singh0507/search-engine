# Search Engine Platform — Operations Handbook

## Architecture

```
Client → Ingress → FastAPI (113 endpoints)
  ├── Search (BM25 + Semantic + Hybrid + Reranking)
  ├── RAG (8 stages: retrieve → cite → ground)
  ├── Agents (5 types + DAG workflows)
  ├── Events (InMemory ↔ Kafka)
  └── Observability (Prometheus + Traces + Logs)
  ↕         ↕          ↕
PostgreSQL  Redis   FAISS/Qdrant
```

## Deployment

| Method | Command |
|--------|---------|
| Local | `python main.py` |
| Docker | `docker-compose up` |
| Docker + Kafka | `docker-compose --profile kafka up` |
| Docker + Qdrant | `docker-compose --profile vector up` |
| Docker + Monitoring | `docker-compose --profile monitoring up` |
| Kubernetes | `kubectl apply -f k8s/` |

## Environment Variables

| Variable | Default | Options |
|----------|---------|---------|
| `DATABASE_BACKEND` | `sqlite` | `sqlite`, `postgres` |
| `EVENT_BACKEND` | `memory` | `memory`, `kafka` |
| `VECTOR_BACKEND` | `faiss` | `faiss`, `qdrant` |
| `CRAWLER_MODE` | `single` | `single`, `distributed` |
| `AGENT_MODE` | `local` | `local`, `distributed` |
| `SECURITY_ENABLED` | `false` | `true`, `false` |
| `TENANCY_ENABLED` | `false` | `true`, `false` |
| `POSTGRES_HOST` | `localhost` | hostname |
| `POSTGRES_PORT` | `5432` | port |
| `REDIS_HOST` | `localhost` | hostname |
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | broker list |
| `JWT_SECRET` | dev secret | production secret |

## Security

Enable: `SECURITY_ENABLED=true`

Roles: `admin` (all), `operator` (index+search), `agent_user` (agents), `reader` (search only)

JWT: `POST /security/token?sub=user&roles=admin` → use `Authorization: Bearer <token>`

## Multi-Tenancy

Enable: `TENANCY_ENABLED=true`

All requests need `X-Tenant-ID` header. Data isolation at document, vector, cache, and session layers.

## Monitoring Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/health` | Liveness |
| `/metrics` | Prometheus |
| `/resilience/health-probes` | Dependency checks |
| `/resilience/circuit-breakers` | Breaker states |
| `/observability/traces` | Traces |
| `/observability/logs` | Structured logs |
| `/cost/summary` | Cost tracking |

## Incident Response

1. `/health` → check liveness
2. `/resilience/health-probes` → check dependencies
3. `/resilience/circuit-breakers` → check breakers
4. `/observability/logs?level=ERROR` → check errors
5. `/events/dlq` → check dead letters

## Backup

- PostgreSQL: `pg_dump` / `psql -f backup.sql`
- FAISS: auto-saved to `data/faiss_index/`
- Redis: appendonly persistence

## Scaling

- HPA: 2-10 pods, CPU 70%
- PDB: minAvailable 1
- FAISS → Qdrant at 1M+ docs

## Runbooks

- Deploy: `kubectl set image deployment/search-engine app=image:tag`
- Rollback: `kubectl rollout undo deployment/search-engine`
- Add tenant: `POST /tenants?tenant_id=new&name=New`
- Rotate JWT: update K8s secret + `kubectl rollout restart`
