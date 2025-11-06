# Architecture Optimization - Visual Diagrams

This document contains visual representations of the current and optimized architectures using Mermaid diagrams.

---

## Current Architecture

### System Overview

```mermaid
graph TB
    CLI[CLI Layer main.py] --> Processor[Article Processor]
    CLI2[CLI Layer main_etl.py] --> Orchestrator[ETL Orchestrator]

    Processor --> RSS[RSS Parser]
    Processor --> Scraper[Web Scraper]
    Processor --> AI[AI Client]
    Processor --> DB1[Database Manager src/core]

    Orchestrator --> Fetcher[Content Fetcher]
    Orchestrator --> Engine[Analysis Engine]
    Orchestrator --> DB2[Database Manager src/etl/load]

    Fetcher --> RSS2[RSS Parser]
    Fetcher --> Scraper2[Web Scraper]

    Engine --> AI2[AI Clients]

    DB1 --> SQLite[(SQLite DB)]
    DB2 --> SQLite

    style CLI fill:#ffcccc
    style CLI2 fill:#ffcccc
    style DB1 fill:#ff9999
    style DB2 fill:#ff9999
    style SQLite fill:#ff6666

    note1[Duplicate Entry Points]
    note2[Duplicate DB Managers]

    note1 -.-> CLI
    note1 -.-> CLI2
    note2 -.-> DB1
    note2 -.-> DB2
```

### Current Sequential Processing Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Processor
    participant RSS
    participant Scraper
    participant AI
    participant DB

    User->>CLI: run --limit 10
    CLI->>Processor: Initialize

    loop For each article (sequential)
        Processor->>RSS: Fetch RSS entry
        RSS-->>Processor: Entry data
        Note over Processor: Wait 200ms

        Processor->>Scraper: Scrape content
        Note over Scraper: HTTP request + parsing
        Scraper-->>Processor: Content (2-3s)
        Note over Processor: Wait 2-3s

        Processor->>AI: Analyze content
        Note over AI: API call + rate limit
        AI-->>Processor: Analysis (3-5s)
        Note over Processor: Wait 3-5s

        Processor->>DB: Insert article
        DB-->>Processor: Article ID
        Processor->>DB: Insert content
        Processor->>DB: Update status
        Processor->>DB: Log processing
        Note over Processor: 4 DB operations
    end

    Processor->>CLI: Results
    CLI->>User: Display report

    Note over User,DB: Total: 8.3s per article × 10 = 83s
```

### Current Performance Bottleneck Analysis

```mermaid
pie title Processing Time Breakdown (per article)
    "AI Analysis" : 57
    "Web Scraping" : 36
    "Database I/O" : 6
    "RSS Parsing" : 4
```

---

## Optimized Architecture (After Implementation)

### Phase 1: Caching + Database Optimization

```mermaid
graph TB
    CLI[Single CLI Entry] --> AsyncOrch[Async ETL Orchestrator]

    AsyncOrch --> CacheMgr[Content Cache Manager]
    AsyncOrch --> Fetcher[Async Content Fetcher]
    AsyncOrch --> Engine[Async Analysis Engine]
    AsyncOrch --> DBPool[Database with Connection Pool]

    CacheMgr --> L1Cache[(L1: Memory Cache<br/>LRU, 100 items)]
    CacheMgr --> L2Cache[(L2: SQLite Cache<br/>Persistent)]

    Fetcher --> RSS[Async RSS Parser]
    Fetcher --> Scraper[Async Web Scraper]

    Scraper -.check cache.-> CacheMgr

    Engine --> AI[Async AI Client]
    Engine --> AICache[AI Result Cache]

    AI -.check cache.-> AICache

    DBPool --> SQLite[(SQLite DB<br/>WAL Mode)]
    DBPool --> BatchOps[Batch Operations]

    style CacheMgr fill:#90EE90
    style L1Cache fill:#98FB98
    style L2Cache fill:#98FB98
    style AICache fill:#98FB98
    style DBPool fill:#87CEEB
    style AsyncOrch fill:#FFD700

    note1[Single Entry Point]
    note2[Multi-Layer Cache]
    note3[Connection Pooling]

    note1 -.-> CLI
    note2 -.-> CacheMgr
    note3 -.-> DBPool
```

### Phase 2: Async/Await Processing Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant AsyncOrch as Async Orchestrator
    participant Semaphore
    participant Cache
    participant Scraper
    participant AI
    participant DB

    User->>CLI: run --limit 10
    CLI->>AsyncOrch: Initialize with concurrency=5

    Note over AsyncOrch: Create async tasks for all articles

    par Article 1
        AsyncOrch->>Semaphore: Acquire (1/5)
        AsyncOrch->>Cache: Check cache
        alt Cache Hit
            Cache-->>AsyncOrch: Cached content (50ms)
        else Cache Miss
            AsyncOrch->>Scraper: Scrape async
            Scraper-->>AsyncOrch: Content (2-3s)
            AsyncOrch->>Cache: Store in cache
        end
        AsyncOrch->>AI: Analyze async
        AI-->>AsyncOrch: Analysis (3-5s)
        AsyncOrch->>Semaphore: Release
    and Article 2
        AsyncOrch->>Semaphore: Acquire (2/5)
        AsyncOrch->>Cache: Check cache
        Cache-->>AsyncOrch: Process...
        AsyncOrch->>AI: Analyze async
        AI-->>AsyncOrch: Analysis
        AsyncOrch->>Semaphore: Release
    and Article 3
        AsyncOrch->>Semaphore: Acquire (3/5)
        AsyncOrch->>Cache: Check cache
        Cache-->>AsyncOrch: Process...
        AsyncOrch->>AI: Analyze async
        AI-->>AsyncOrch: Analysis
        AsyncOrch->>Semaphore: Release
    and Article 4
        AsyncOrch->>Semaphore: Acquire (4/5)
        AsyncOrch->>Cache: Check cache
        Cache-->>AsyncOrch: Process...
        AsyncOrch->>AI: Analyze async
        AI-->>AsyncOrch: Analysis
        AsyncOrch->>Semaphore: Release
    and Article 5
        AsyncOrch->>Semaphore: Acquire (5/5)
        AsyncOrch->>Cache: Check cache
        Cache-->>AsyncOrch: Process...
        AsyncOrch->>AI: Analyze async
        AI-->>AsyncOrch: Analysis
        AsyncOrch->>Semaphore: Release
    end

    Note over AsyncOrch: Batch store results
    AsyncOrch->>DB: Batch insert (single transaction)
    DB-->>AsyncOrch: Article IDs

    AsyncOrch->>CLI: Results
    CLI->>User: Display report

    Note over User,DB: Total: 21s for 10 articles (vs 83s sequential)<br/>75% faster!
```

### Caching Layer Architecture

```mermaid
graph LR
    Request[Incoming Request] --> L1{L1: Memory Cache}

    L1 -->|Hit 40-50%| Return1[Return<br/>~100ms]
    L1 -->|Miss| L2{L2: SQLite Cache}

    L2 -->|Hit 30-40%| Store1[Store in L1]
    Store1 --> Return2[Return<br/>~20ms]

    L2 -->|Miss| L3[L3: Web Scraping]
    L3 --> Store2[Store in L2]
    Store2 --> Store3[Store in L1]
    Store3 --> Return3[Return<br/>~3000ms]

    style L1 fill:#90EE90
    style L2 fill:#87CEEB
    style L3 fill:#FFB6C1
    style Return1 fill:#98FB98
    style Return2 fill:#98FB98
    style Return3 fill:#FF6B6B
```

---

## Performance Comparison

### Processing Time Comparison

```mermaid
gantt
    title Article Processing Time Comparison
    dateFormat X
    axisFormat %Ls

    section Current Sequential
    Article 1 :0, 8300ms
    Article 2 :8300ms, 8300ms
    Article 3 :16600ms, 8300ms

    section Optimized Async (5 concurrent)
    Article 1 :0, 8300ms
    Article 2 :0, 8300ms
    Article 3 :0, 8300ms
    Article 4 :8300ms, 8300ms
    Article 5 :8300ms, 8300ms
    Article 6 :8300ms, 8300ms
```

### Throughput Improvement

```mermaid
graph LR
    subgraph "Current (Sequential)"
        A1[6 articles/min]
    end

    subgraph "Phase 1 (Cached)"
        B1[15 articles/min]
        B2[+150%]
    end

    subgraph "Phase 2 (Async)"
        C1[25 articles/min]
        C2[+317%]
    end

    A1 -->|Add Caching| B1
    B1 -->|Add Async| C1

    style A1 fill:#FFB6C1
    style B1 fill:#FFD700
    style C1 fill:#90EE90
```

### Cost Reduction

```mermaid
graph TB
    subgraph "Current Monthly Costs"
        A1[API Calls: $120]
        A2[Compute: $30]
        A3[Total: $150/month]
    end

    subgraph "After Phase 1 (Caching)"
        B1[API Calls: $48]
        B2[60% reduction]
        B3[Compute: $25]
        B4[Total: $73/month]
    end

    subgraph "After Phase 2 (Async)"
        C1[API Calls: $36]
        C2[70% reduction]
        C3[Compute: $20]
        C4[Total: $56/month]
    end

    A3 -->|Implement Caching| B4
    B4 -->|Implement Async| C4

    style A3 fill:#FF6B6B
    style B4 fill:#FFD700
    style C4 fill:#90EE90
```

---

## Database Optimization

### Current: Individual Operations

```mermaid
sequenceDiagram
    participant Processor
    participant DB

    Note over Processor,DB: For each article (sequential)

    loop 10 articles
        Processor->>DB: Connect
        Processor->>DB: INSERT article
        DB-->>Processor: article_id
        Processor->>DB: Disconnect

        Processor->>DB: Connect
        Processor->>DB: INSERT content
        DB-->>Processor: content_id
        Processor->>DB: Disconnect

        Processor->>DB: Connect
        Processor->>DB: UPDATE article status
        DB-->>Processor: OK
        Processor->>DB: Disconnect

        Processor->>DB: Connect
        Processor->>DB: INSERT processing_log
        DB-->>Processor: log_id
        Processor->>DB: Disconnect
    end

    Note over Processor,DB: 40 connections, 40 operations<br/>Total: 2000ms
```

### Optimized: Batch Operations with Connection Pool

```mermaid
sequenceDiagram
    participant Processor
    participant Pool as Connection Pool
    participant DB

    Processor->>Pool: Get connection (thread-local)
    Pool-->>Processor: Reusable connection

    Note over Processor,DB: Single transaction for batch

    Processor->>DB: BEGIN TRANSACTION
    Processor->>DB: INSERT articles (batch of 10)
    DB-->>Processor: article_ids[10]
    Processor->>DB: INSERT content (batch of 10)
    DB-->>Processor: content_ids[10]
    Processor->>DB: INSERT logs (batch of 10)
    DB-->>Processor: log_ids[10]
    Processor->>DB: COMMIT

    Note over Pool: Connection stays in pool

    Note over Processor,DB: 1 connection, 3 batch operations<br/>Total: 100ms (20x faster!)
```

### Connection Pool Architecture

```mermaid
graph TB
    subgraph "Thread-Local Connection Pool"
        Thread1[Thread 1] --> Conn1[SQLite Connection 1]
        Thread2[Thread 2] --> Conn2[SQLite Connection 2]
        Thread3[Thread 3] --> Conn3[SQLite Connection 3]
        Thread4[Thread 4] --> Conn4[SQLite Connection 4]
    end

    Conn1 --> SQLite[(SQLite DB<br/>WAL Mode)]
    Conn2 --> SQLite
    Conn3 --> SQLite
    Conn4 --> SQLite

    style Thread1 fill:#FFD700
    style Thread2 fill:#FFD700
    style Thread3 fill:#FFD700
    style Thread4 fill:#FFD700
    style SQLite fill:#87CEEB

    Note1[Each thread reuses<br/>its own connection]
    Note1 -.-> Conn1
```

---

## Async/Await Execution Flow

### CPU Utilization Comparison

```mermaid
gantt
    title CPU Utilization Over Time
    dateFormat X
    axisFormat %Ls

    section Current (Sequential)
    CPU Active   :active, 0, 500ms
    Waiting I/O  :crit, 500ms, 7800ms
    CPU Active   :active, 8300ms, 500ms
    Waiting I/O  :crit, 8800ms, 7800ms

    section Optimized (Async 5x)
    CPU 1 Active :active, 0, 500ms
    CPU 2 Active :active, 0, 500ms
    CPU 3 Active :active, 0, 500ms
    CPU 4 Active :active, 0, 500ms
    CPU 5 Active :active, 0, 500ms
    Waiting I/O  :crit, 500ms, 3000ms
```

### Concurrency Control with Semaphore

```mermaid
graph TB
    Start[10 Articles to Process] --> Queue[Task Queue]

    Queue --> Semaphore{Semaphore<br/>Limit: 5}

    Semaphore -->|Slot 1| Task1[Task 1<br/>Article 1]
    Semaphore -->|Slot 2| Task2[Task 2<br/>Article 2]
    Semaphore -->|Slot 3| Task3[Task 3<br/>Article 3]
    Semaphore -->|Slot 4| Task4[Task 4<br/>Article 4]
    Semaphore -->|Slot 5| Task5[Task 5<br/>Article 5]
    Semaphore -.waiting.-> Wait1[Article 6-10<br/>Waiting]

    Task1 -->|Complete| Release1[Release Slot]
    Task2 -->|Complete| Release2[Release Slot]
    Task3 -->|Complete| Release3[Release Slot]
    Task4 -->|Complete| Release4[Release Slot]
    Task5 -->|Complete| Release5[Release Slot]

    Release1 -.-> Wait1
    Release2 -.-> Wait1

    Wait1 --> Next[Next 5 Articles<br/>Start Processing]

    style Semaphore fill:#FFD700
    style Task1 fill:#90EE90
    style Task2 fill:#90EE90
    style Task3 fill:#90EE90
    style Task4 fill:#90EE90
    style Task5 fill:#90EE90
    style Wait1 fill:#FFB6C1
```

---

## Implementation Phases

### Phase 1: Quick Wins (2 weeks)

```mermaid
graph LR
    Start[Start] --> Week1[Week 1: Caching]
    Week1 --> Cache1[Content Cache Manager]
    Week1 --> Cache2[AI Result Cache]
    Week1 --> Cache3[Database Cache Tables]

    Week1 --> Week2[Week 2: DB Optimization]
    Week2 --> DB1[Connection Pooling]
    Week2 --> DB2[Batch Operations]
    Week2 --> DB3[WAL Mode]

    Week2 --> Results1[Results]

    Results1 --> R1[50-70% Cost Reduction]
    Results1 --> R2[40-60% DB Performance]
    Results1 --> R3[Test Coverage]

    style Cache1 fill:#90EE90
    style Cache2 fill:#90EE90
    style Cache3 fill:#90EE90
    style DB1 fill:#87CEEB
    style DB2 fill:#87CEEB
    style DB3 fill:#87CEEB
    style Results1 fill:#FFD700
```

### Phase 2: Async Pipeline (4 weeks)

```mermaid
graph LR
    Week3[Week 3: Async Foundation] --> AI1[Async AI Client]
    Week3 --> AI2[aiohttp Integration]

    Week4[Week 4: Async Scraping] --> Scraper1[Async Web Scraper]
    Week4 --> Scraper2[Async RSS Parser]

    Week5[Week 5: Orchestrator] --> Orch1[Async ETL Orchestrator]
    Week5 --> Orch2[Concurrent Processing]

    Week6[Week 6: Testing] --> Test1[Integration Tests]
    Week6 --> Test2[Load Tests]
    Week6 --> Test3[Performance Tests]

    AI1 --> Week4
    AI2 --> Week4
    Scraper1 --> Week5
    Scraper2 --> Week5
    Orch1 --> Week6
    Orch2 --> Week6

    Week6 --> Results2[Results]

    Results2 --> R4[60-75% Time Reduction]
    Results2 --> R5[3-4x Throughput]
    Results2 --> R6[Backward Compatible]

    style Week3 fill:#FFB6C1
    style Week4 fill:#FFB6C1
    style Week5 fill:#FFB6C1
    style Week6 fill:#FFB6C1
    style Results2 fill:#FFD700
```

### Phase 3: Code Consolidation (1 week)

```mermaid
graph LR
    Week7[Week 7: Cleanup] --> Clean1[Remove Duplicate DB]
    Week7 --> Clean2[Consolidate Entry Points]
    Week7 --> Clean3[Create Utilities]

    Clean1 --> Results3[Results]
    Clean2 --> Results3
    Clean3 --> Results3

    Results3 --> R7[30% Code Reduction]
    Results3 --> R8[Simplified Architecture]
    Results3 --> R9[Better Maintainability]

    style Week7 fill:#98FB98
    style Results3 fill:#FFD700
```

---

## Scalability Roadmap

### Current vs Future Scaling

```mermaid
graph TB
    subgraph "Current (Single Server)"
        Current[Single Process<br/>SQLite<br/>~100 articles/day]
    end

    subgraph "Phase 1-3 (Optimized Single)"
        P123[Async + Caching<br/>SQLite + WAL<br/>~1000 articles/day]
    end

    subgraph "Future: Distributed (Optional)"
        Redis[Redis Task Queue]
        Worker1[Worker 1]
        Worker2[Worker 2]
        Worker3[Worker 3]
        Postgres[(PostgreSQL)]

        Redis --> Worker1
        Redis --> Worker2
        Redis --> Worker3

        Worker1 --> Postgres
        Worker2 --> Postgres
        Worker3 --> Postgres
    end

    Current -->|Implement<br/>Optimizations| P123
    P123 -.->|If needed<br/>>5000 articles/day| Redis

    style Current fill:#FFB6C1
    style P123 fill:#90EE90
    style Redis fill:#FFD700
```

---

## Key Metrics Dashboard

### Expected Improvements Summary

```mermaid
graph TB
    subgraph "Processing Time"
        T1[Current: 82.5s<br/>for 10 articles]
        T2[Phase 1: 41s<br/>50% faster]
        T3[Phase 2: 21s<br/>75% faster]
    end

    subgraph "API Costs"
        C1[Current: $120/month]
        C2[Phase 1: $48/month<br/>60% reduction]
        C3[Phase 2: $36/month<br/>70% reduction]
    end

    subgraph "Throughput"
        Th1[Current: 6 art/min]
        Th2[Phase 1: 15 art/min<br/>+150%]
        Th3[Phase 2: 25 art/min<br/>+317%]
    end

    T1 --> T2 --> T3
    C1 --> C2 --> C3
    Th1 --> Th2 --> Th3

    style T1 fill:#FFB6C1
    style T2 fill:#FFD700
    style T3 fill:#90EE90
    style C1 fill:#FFB6C1
    style C2 fill:#FFD700
    style C3 fill:#90EE90
    style Th1 fill:#FFB6C1
    style Th2 fill:#FFD700
    style Th3 fill:#90EE90
```

---

## Legend

- 🔴 **Red/Pink** - Current issues, bottlenecks, problems
- 🟡 **Yellow/Gold** - Work in progress, medium priority
- 🟢 **Green** - Optimized, implemented, success
- 🔵 **Blue** - Database, storage, persistence
- ⚪ **Gray** - Future enhancements, optional

---

**Document Status:** ✅ Complete
**Last Updated:** 2025-10-12
**Version:** 1.0
