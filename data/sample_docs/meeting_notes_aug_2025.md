# Meeting Notes — Engineering All-Hands, August 28, 2025

**Date:** August 28, 2025
**Location:** Remote (Zoom)
**Attendees:** All Engineering (~42 people)

---

## Agenda

1. Q3 progress update
2. Product roadmap preview
3. Hiring update
4. Open Q&A

---

## Q3 Progress Update (Priya Menon)

- Data platform migration is 90% complete; on track for September 5th finish
- API reliability improved to 99.97% uptime (up from 99.91% in Q2)
- Three P1 incidents in Q3 (down from seven in Q2); incident review process working

## Product Roadmap Preview (Ravi Krishnan, CPO)

Q4 focus areas:
1. **Document Intelligence API** — public beta in November 2025
2. **Multi-tenant workspace isolation** — enterprise tier
3. **Webhook support** for async job notifications

No major breaking changes planned for Q4. API versioning policy will be formalized in October.

## Hiring Update

Currently open roles:
- 2x Backend Engineer (L3–L4), team: Platform
- 1x ML Engineer (L4–L5), team: AI Research
- 1x DevOps / SRE (L3), team: Infrastructure

Interview process: take-home assignment + 2 technical + 1 system design + 1 culture.

## Changelog — Recent Releases

### v2.4.0 (August 20, 2025)
- Added PDF chunking with semantic boundaries
- Improved embedding cache hit rate by 34%
- Fixed: race condition in concurrent ingestion jobs

### v2.3.1 (August 15, 2025)
- Hot patch: memory leak fix in ingestion service (caused August 14th P1)

### v2.3.0 (August 1, 2025)
- DOCX parser now extracts tables as Markdown
- Added BM25 keyword search as hybrid retrieval fallback

## Open Q&A Notes

**Q (Marcus Webb):** Will we support streaming responses for the Document Intelligence API?
**A (Ravi):** Yes, SSE streaming is on the roadmap for Q1 2026.

**Q (Anonymous):** Is the take-home assignment paid?
**A (Priya):** Currently no, but we limit it to 3-4 hours. We're reviewing this policy.

---

*Notes by: Keiko Tanaka*
