# RLM Data Flow Diagrams

Visual diagrams for the RLM (Recursive Language Model) and CausalFrame system.

## Overview Diagram

```mermaid
flowchart TB
    subgraph MainLLM["Main LLM (Claude Code)"]
        User[User Query]
        MainResponse[Final Response]
    end

    subgraph RLM["RLM System"]
        REPL[REPL Environment]
        FrameIndex[FrameIndex]
        FrameStore[FrameStore]
    end

    subgraph Storage["Persistent Storage"]
        IndexFile[index.json]
        FramesDir[frames.jsonl]
        SessionDir[Session Artifacts]
    end

    User -->|"1. Query"| REPL
    REPL -->|"2. Search"| FrameIndex
    FrameIndex -->|"3. Load"| FrameStore
    FrameStore -->|"4. Return Frames"| REPL
    REPL -->|"5. Execute Code"| PythonExec[Python Execution]
    PythonExec -->|"6. Create Frame"| FrameStore
    FrameStore -->|"7. Save"| FramesDir
    FrameStore -->|"8. Index"| IndexFile
    REPL -->|"9. Results"| MainResponse
```

## Phase 1-8: Complete Data Flow

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant M as Main LLM
    participant R as RLM/REPL
    participant FI as FrameIndex
    participant FS as FrameStore
    participant D as Disk Storage

    Note over U,D: Phase 1-2: Initialization & Request
    U->>M: Send query
    M->>M: Determine if RLM needed

    Note over M,D: Phase 3: RLM Invocation
    M->>R: Invoke RLM with task
    R->>FI: Initialize/load index

    Note over R,D: Phase 4: Frame Retrieval
    R->>FI: Search for relevant frames
    FI->>FS: Find matching frame IDs
    FS->>D: Load frame data
    D-->>FS: Frame JSON
    FS-->>R: CausalFrame objects

    Note over R,D: Phase 5: Code Execution
    R->>R: Execute Python code
    R->>R: Access files via REPL

    Note over R,D: Phase 6: Frame Creation
    R->>R: Create new CausalFrame
    R->>FS: Save frame
    FS->>D: Write to frames.jsonl
    R->>FI: Index new frame
    FI->>D: Update index.json

    Note over R,D: Phase 7-8: Response
    R-->>M: Return results + frame refs
    M->>M: Synthesize response
    M->>U: Final response
```

## Frame Persistence Process

```mermaid
flowchart LR
    subgraph Creation["Frame Creation"]
        A1[1. Instantiate<br/>CausalFrame] --> A2[2. Attach<br/>Metadata]
        A2 --> A3[3. Create<br/>ContextSlices]
        A3 --> A4[4. Serialize<br/>to JSON]
    end

    subgraph Storage["Storage"]
        A4 --> B1[5. Save to<br/>FrameStore]
        B1 --> B2[6. Add to<br/>FrameIndex]
    end

    subgraph Disk["Disk"]
        B2 --> C1[frames.jsonl]
        B2 --> C2[index.json]
    end

    style A1 fill:#e1f5fe
    style B1 fill:#fff3e0
    style C1 fill:#e8f5e9
```

## CausalFrame Structure

```mermaid
classDiagram
    class CausalFrame {
        +String id
        +String session_id
        +DateTime created_at
        +String frame_type
        +String content
        +Dict metadata
        +List~ContextSlice~ context_slices
        +List~String~ source_files
        +List~String~ dependencies
        +List~String~ tags
        +FrameStatus status
        +to_dict() Dict
        +to_json() String
    }

    class ContextSlice {
        +String file_path
        +String content
        +Tuple line_range
        +String content_hash
    }

    class FrameStatus {
        <<enumeration>>
        RUNNING
        COMPLETED
        SUSPENDED
        INVALIDATED
        PROMOTED
    }

    CausalFrame "1" --> "*" ContextSlice : contains
    CausalFrame --> FrameStatus : has status
```

## Loading Old Frames

```mermaid
flowchart TD
    Start[Start: Need Old Frames] --> LoadIndex[FrameIndex.load]
    LoadIndex --> IndexLoaded{Index Loaded?}

    IndexLoaded -->|Yes| Search[FrameIndex.search query]
    IndexLoaded -->|No| Empty[Return Empty]

    Search --> Results{Found Frames?}
    Results -->|Yes| LoadFrames[FrameStore.load_frame ids]
    Results -->|No| Empty

    LoadFrames --> SessionCheck{Cross-session?}
    SessionCheck -->|Yes| LoadSession[SessionComparison.load_session]
    SessionCheck -->|No| Return

    LoadSession --> Merge[Merge with current]
    Merge --> Return[Return CausalFrames]

    style Start fill:#e3f2fd
    style Return fill:#e8f5e9
    style Empty fill:#fff8e1
```

## Loading Mechanisms Comparison

```mermaid
flowchart LR
    subgraph Methods["Loading Methods"]
        M1[FrameIndex.load]
        M2[FrameStore.load_frame]
        M3[SessionComparison.load_session]
        M4[FrameIndex.search]
    end

    subgraph UseCases["Use Cases"]
        U1[Get all frames metadata]
        U2[Load specific frame by ID]
        U3[Load all frames from session]
        U4[Search frames by pattern]
    end

    M1 -.->|returns| U1
    M2 -.->|returns| U2
    M3 -.->|returns| U3
    M4 -.->|returns| U4
```

## Invalidation Chain Cascade

```mermaid
flowchart TD
    subgraph Trigger["Trigger"]
        FileChange[File Modified]
    end

    subgraph Detection["Detection"]
        FileChange --> FindFrames[FrameIndex.find_frames_by_file]
        FindFrames --> DirectFrames[Frames with source_files]
    end

    subgraph Cascade["Cascade Invalidation"]
        DirectFrames --> MarkInvalid1[Mark INVALID]
        MarkInvalid1 --> TraverseDeps[Traverse dependencies]
        TraverseDeps --> DependentFrames[Find dependent frames]
        DependentFrames --> MarkInvalid2[Mark INVALID]
        MarkInvalid2 --> MoreDeps{More deps?}
        MoreDeps -->|Yes| TraverseDeps
        MoreDeps -->|No| UpdateIndex[Update FrameIndex]
    end

    subgraph Result["Result"]
        UpdateIndex --> InvalidFrames[Invalidated Frames List]
        InvalidFrames --> Revalidation[RLM Re-executes Analysis]
        Revalidation --> NewFrame[New Valid Frame]
    end

    style FileChange fill:#ffcdd2
    style MarkInvalid1 fill:#ffcdd2
    style MarkInvalid2 fill:#ffcdd2
    style NewFrame fill:#c8e6c9
```

## Invalidation Rules

```mermaid
flowchart TB
    subgraph Rules["Invalidation Rules"]
        R1[Rule 1: Direct File Reference]
        R2[Rule 2: Dependency Chain]
        R3[Rule 3: Context Slice Mismatch]
    end

    subgraph Conditions["Conditions"]
        C1[frame.source_files<br/>contains changed_file]
        C2[frame.dependencies<br/>contains invalid_frame_id]
        C3[ContextSlice.content<br/>≠ current file content]
    end

    subgraph Actions["Actions"]
        A[Mark Frame INVALID]
    end

    R1 --> C1 --> A
    R2 --> C2 --> A
    R3 --> C3 --> A

    style A fill:#ffcdd2
```

## Dependency Graph Example

```mermaid
graph TD
    subgraph Session1["Session 1"]
        F1[Frame A<br/>source: auth.py]
        F2[Frame B<br/>source: login.py]
    end

    subgraph Session2["Session 2"]
        F3[Frame C<br/>depends: A, B]
        F4[Frame D<br/>depends: C]
    end

    subgraph Session3["Session 3"]
        F5[Frame E<br/>depends: D]
    end

    F1 --> F3
    F2 --> F3
    F3 --> F4
    F4 --> F5

    F1 -.->|INVALIDATED| X1[x]
    F3 -.->|CASCADE| X2[x]
    F4 -.->|CASCADE| X3[x]
    F5 -.->|CASCADE| X4[x]

    style F1 fill:#ffcdd2
    style F3 fill:#ffcdd2
    style F4 fill:#ffcdd2
    style F5 fill:#ffcdd2
    style F2 fill:#c8e6c9
```

## Session Artifacts Structure

```mermaid
flowchart TB
    subgraph Storage["~/.claude/rlm-frames/"]
        Coord[.current_session]
        S1[session_id_1/]
        S2[session_id_2/]
    end

    subgraph SessionFolder["Session Folder"]
        Index[index.json]
        Frames[frames.jsonl]
        Tools[tools.jsonl]
        Artifacts[artifacts.json]
    end

    S1 --> SessionFolder
    S2 --> SessionFolder2[Same Structure]

    subgraph ArtifactsContent["artifacts.json"]
        A1[session_id]
        A2[created_at]
        A3[frames_created]
        A4[files_accessed]
        A5[parent_sessions]
    end

    Artifacts --> ArtifactsContent

    style Storage fill:#e3f2fd
    style SessionFolder fill:#fff3e0
```

## Revalidation Process

```mermaid
sequenceDiagram
    autonumber
    participant R as RLM
    participant FI as FrameIndex
    participant FS as FrameStore
    participant D as Disk

    Note over R,D: Detection Phase
    R->>FI: Check for invalid frames
    FI-->>R: Return invalid frame list

    Note over R,D: Revalidation Phase
    loop For each invalid frame
        R->>FS: Load original frame
        FS-->>R: Frame with query + context
        R->>R: Re-execute analysis
        R->>R: Create new frame with updated content
        R->>FS: Save new frame
        R->>FI: Archive old frame
        FI->>D: Update index
    end

    Note over R,D: Complete
    R-->>R: All frames revalidated
```
