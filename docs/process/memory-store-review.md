# MemoryStore Architecture & CausalFrame Integration Review

## Executive Summary

This document provides a comprehensive review of the MemoryStore architecture and its integration with CausalFrame (SPEC-17). The MemoryStore serves as the persistence layer for conversational context, while CausalFrame provides a structured approach to tracking causal relationships and context evolution across conversations.

---

## 1. MemoryStore Architecture

### 1.1 Purpose & Responsibilities

The **MemoryStore** is the central persistence layer responsible for:
- Storing and retrieving conversational context
- Managing frame lifecycle (creation, updates, archival)
- Providing query interfaces for context retrieval
- Handling persistence across sessions

### 1.2 Core Components

#### 1.2.1 MemoryStore Class
**Location:** `memory_store.py`

**Key Attributes:**
- `backend`: Abstract storage backend (MemoryBackend interface)
- `index`: FrameIndex for efficient lookups
- `lifecycle_manager`: FrameLifecycle for state transitions
- `serializer`: FrameSerialization for data persistence

**Key Methods:**
```python
class MemoryStore:
    def store_frame(self, frame: CausalFrame) -> str
    def retrieve_frame(self, frame_id: str) -> Optional[CausalFrame]
    def query_context(self, query: ContextQuery) -> List[CausalFrame]
    def evolve_frame(self, frame_id: str, evolution: FrameEvolution) -> CausalFrame
    def archive_frame(self, frame_id: str) -> bool
    def get_active_frames(self, session_id: str) -> List[CausalFrame]
```

### 1.3 Data Persistence

#### 1.3.1 MemoryBackend Interface
**Location:** `memory_backend.py`

The backend abstraction allows pluggable storage implementations:
- **FileBackend**: JSON-based file storage
- **DatabaseBackend**: SQL/NoSQL database storage
- **HybridBackend**: Multi-tier caching strategy

**Interface Methods:**
```python
class MemoryBackend(ABC):
    @abstractmethod
    def save(self, frame_id: str, data: dict) -> bool

    @abstractmethod
    def load(self, frame_id: str) -> Optional[dict]

    @abstractmethod
    def delete(self, frame_id: str) -> bool

    @abstractmethod
    def query(self, filters: dict) -> List[dict]
```

#### 1.3.2 FrameSerialization
**Location:** `frame_serialization.py`

Handles conversion between CausalFrame objects and persistent storage format:
- **Serialization**: CausalFrame → JSON/Protobuf/Binary
- **Deserialization**: Storage format → CausalFrame
- **Compression**: Optional compression for large frames
- **Encryption**: Optional encryption for sensitive data

---

## 2. CausalFrame Architecture

### 2.1 Problem Statement

CausalFrame addresses the challenge of:
1. **Context Tracking**: Maintaining coherent context across conversation turns
2. **Causal Relationships**: Understanding how earlier context influences later responses
3. **Context Evolution**: Tracking how context changes over time
4. **Reference Resolution**: Linking mentions to their referents

### 2.2 Data Structure

**Location:** `causal_frame.py`

```python
@dataclass
class CausalFrame:
    # Identity
    frame_id: str
    session_id: str
    parent_frame_id: Optional[str]  # For causal chain tracking

    # Content
    content: FrameContent
    context: ContextWindow

    # Metadata
    timestamp: datetime
    frame_type: FrameType
    lifecycle_state: LifecycleState

    # Causal Tracking
    causal_links: List[CausalLink]
    dependencies: List[str]  # IDs of frames this depends on

    # Evolution
    evolution_history: List[FrameEvolution]
    version: int
```

### 2.3 Core Components

#### 2.3.1 FrameContent
The actual conversational content:
- `messages`: List of messages in this frame
- `entities`: Extracted entities/references
- `summaries`: Compressed representations
- `embeddings`: Vector representations for semantic search

#### 2.3.2 ContextWindow
Sliding window of relevant context:
- `active_context`: Currently relevant information
- `context_bounds`: Start/end positions in conversation
- `attention_weights`: Importance scores for context elements

#### 2.3.3 CausalLink
Represents causal relationships:
```python
@dataclass
class CausalLink:
    source_frame_id: str
    target_frame_id: str
    link_type: CausalLinkType  # REFERENCE, ELABORATION, CORRECTION, etc.
    strength: float  # 0.0 to 1.0
    metadata: dict
```

### 2.4 Frame Lifecycle

**Location:** `frame_lifecycle.py`

Frames progress through states:
```
CREATED → ACTIVE → EVOLVING → STALE → ARCHIVED
         ↓         ↓
    PINNED  LOCKED
```

**State Transitions:**
- **CREATED**: Initial frame creation
- **ACTIVE**: Currently in use for context
- **EVOLVING**: Being updated with new information
- **STALE**: No longer actively referenced
- **ARCHIVED**: Persisted for long-term storage
- **PINNED**: Prevented from archival (important context)
- **LOCKED**: Read-only, cannot be modified

---

## 3. MemoryStore ↔ CausalFrame Integration (SPEC-17)

### 3.1 Architectural Relationship

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                    │
│                  (Session Manager, etc.)                │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                     MemoryStore                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Frame      │  │    Frame     │  │    Frame     │  │
│  │   Index      │  │   Lifecycle  │  │ Serialization│  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  Memory Backend                         │
│  (File/Database/Hybrid - Pluggable)                     │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Key Integration Points

#### 3.2.1 Storage Operations
MemoryStore stores CausalFrames as the primary unit of persistence:

```python
# Storing a new frame
frame = CausalFrame(
    frame_id="frame_001",
    session_id="session_123",
    content=FrameContent(...),
    context=ContextWindow(...),
    # ... other fields
)
frame_id = memory_store.store_frame(frame)

# Retrieving a frame
retrieved_frame = memory_store.retrieve_frame("frame_001")
```

#### 3.2.2 Context Queries
MemoryStore provides sophisticated querying over CausalFrames:

```python
# Query by causal relationships
results = memory_store.query_context(
    ContextQuery(
        session_id="session_123",
        causal_link_type=CausalLinkType.REFERENCE,
        time_range=(start, end),
        max_depth=3  # Follow causal chains
    )
)
```

#### 3.2.3 Frame Evolution
MemoryStore tracks how frames evolve:

```python
# Evolve a frame with new information
evolution = FrameEvolution(
    evolution_type=EvolutionType.ELABORATION,
    changes=[...],
    timestamp=datetime.now()
)
updated_frame = memory_store.evolve_frame("frame_001", evolution)
```

### 3.3 SPEC-17 Implementation Details

**SPEC-17** defines the integration contract:

1. **Storage Contract**: All frames must be CausalFrame instances
2. **Indexing Requirements**: Index must support causal link traversal
3. **Query Interface**: Must support causal relationship queries
4. **Evolution Tracking**: All frame changes must be recorded in evolution_history
5. **Lifecycle Management**: Frames must respect lifecycle states

---

## 4. Supporting Components

### 4.1 FrameIndex
**Location:** `frame_index.py`

Provides efficient lookup:
- **ID Index**: O(1) frame lookup by ID
- **Session Index**: All frames for a session
- **Causal Index**: Frames linked by causal relationships
- **Temporal Index**: Time-based queries
- **Semantic Index**: Vector similarity search

### 4.2 MemoryEvolution
**Location:** `memory_evolution.py`

Tracks how memory changes:
- **Evolution Types**: ELABORATION, CORRECTION, MERGE, SPLIT
- **Change Detection**: Identifies what changed between versions
- **Conflict Resolution**: Handles concurrent modifications
- **Compression**: Summarizes old frames to save space

### 4.3 SessionManager
**Location:** `session_manager.py`

Manages conversation sessions:
- **Session Creation**: Initialize new conversation contexts
- **Frame Association**: Link frames to sessions
- **Context Window**: Manage sliding window of active frames
- **Session State**: Track session-level metadata

### 4.4 SessionSchema
**Location:** `session_schema.py`

Defines session structure:
```python
@dataclass
class Session:
    session_id: str
    created_at: datetime
    updated_at: datetime
    active_frame_ids: List[str]
    archived_frame_ids: List[str]
    metadata: SessionMetadata
```

---

## 5. Data Flow Examples

### 5.1 Creating a New Frame

```
1. Application creates CausalFrame with content
2. SessionManager assigns session_id and initial context
3. MemoryStore.store_frame() is called
4. FrameLifecycle transitions: CREATED → ACTIVE
5. FrameSerialization converts to storage format
6. MemoryBackend persists the data
7. FrameIndex updates all indices
8. frame_id is returned to application
```

### 5.2 Querying Context

```
1. Application requests context with query
2. MemoryStore.query_context() is called
3. FrameIndex filters candidates by session/time
4. Causal links are traversed to build context chain
5. Frames are deserialized from storage
6. Results are ranked by relevance/strength
7. Ordered list of CausalFrames is returned
```

### 5.3 Evolving a Frame

```
1. Application requests frame evolution
2. MemoryEvolution calculates changes
3. New version is created with incremented version number
4. Evolution is added to evolution_history
5. FrameLifecycle transitions: ACTIVE → EVOLVING → ACTIVE
6. Updated frame is persisted
7. FrameIndex updates affected indices
8. Dependencies are updated if causal links changed
```

---

## 6. Design Patterns & Principles

### 6.1 Patterns Used

1. **Repository Pattern**: MemoryStore abstracts storage details
2. **Strategy Pattern**: Pluggable backend implementations
3. **Observer Pattern**: Lifecycle state change notifications
4. **Command Pattern**: FrameEvolution as reversible operations
5. **Decorator Pattern**: Serialization/compression/encryption layers

### 6.2 Design Principles

1. **Separation of Concerns**: Each module has single responsibility
2. **Interface Segregation**: Small, focused interfaces
3. **Dependency Inversion**: Depend on abstractions, not concretions
4. **Open/Closed**: Open for extension (new backends), closed for modification
5. **Single Source of Truth**: Frame ID is canonical identifier

---

## 7. Performance Considerations

### 7.1 Optimization Strategies

1. **Lazy Loading**: Frames loaded only when needed
2. **Index Caching**: Frequently accessed indices kept in memory
3. **Batch Operations**: Bulk store/reduce for efficiency
4. **Compression**: Old frames compressed to save space
5. **Async Operations**: Non-blocking I/O for large operations

### 7.2 Scalability

1. **Horizontal Scaling**: Distributed backends for large deployments
2. **Sharding**: Session-based sharding for multi-tenant systems
3. **TTL Policies**: Automatic archival of old frames
4. **Connection Pooling**: Efficient database connection management

---

## 8. Security & Privacy

### 8.1 Data Protection

1. **Encryption**: Optional frame-level encryption
2. **Access Control**: Session-level access permissions
3. **Audit Logging**: All frame modifications logged
4. **PII Handling**: Special handling for personally identifiable information

### 8.2 Privacy Features

1. **Ephemeral Frames**: Auto-delete after TTL
2. **Session Isolation**: Frames cannot cross session boundaries
3. **Selective Persistence**: Control which frames are stored
4. **Right to be Forgotten**: Complete frame deletion capability

---

## 9. Future Enhancements

### 9.1 Potential Improvements

1. **Distributed Memory**: Multi-node memory sharing
2. **Semantic Clustering**: Group similar frames automatically
3. **Predictive Prefetching**: Anticipate needed frames
4. **Cross-Session Learning**: Transfer learning between sessions
5. **Hierarchical Frames**: Nested frame structures

### 9.2 Research Directions

1. **Causal Inference**: Better causal relationship detection
2. **Memory Consolidation**: AI-driven frame summarization
3. **Attention Mechanisms**: Dynamic context weighting
4. **Meta-Learning**: Learn optimal retention policies

---

## 10. Conclusion

The MemoryStore architecture with CausalFrame integration (SPEC-17) provides a robust, scalable foundation for conversational context management. The separation of concerns between storage (MemoryStore) and context structure (CausalFrame) enables:

- **Flexible Persistence**: Pluggable backends for different deployment scenarios
- **Rich Context Tracking**: Causal relationships capture conversation flow
- **Efficient Querying**: Multiple indices support diverse access patterns
- **Evolution Tracking**: Complete audit trail of context changes

This architecture is well-suited for applications requiring sophisticated context management, such as AI assistants, collaborative tools, and long-running conversations.

---

**Document Version:** 1.0
**Last Updated:** 2026-02-19
**Specification:** SPEC-17
