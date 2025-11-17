# QCA_Utils.py Refactoring - Phase 1 Checklist

## Pre-Flight Checks
- [ ] Backup original `QCA_Utils.py` (git already has it)
- [ ] Understand current class layout (15 classes identified)
- [ ] Review imports in all affected files
- [ ] Ensure all tests pass before starting

## Phase 1: Directory Structure Setup (1 hour)

### Step 1.1: Create utils directory structure
```bash
mkdir -p QCA_AID_assets/utils/{llm,config,tracking,dialog,export,io}
```

- [ ] Create `QCA_AID_assets/utils/` directory
- [ ] Create `QCA_AID_assets/utils/llm/` directory
- [ ] Create `QCA_AID_assets/utils/config/` directory
- [ ] Create `QCA_AID_assets/utils/tracking/` directory
- [ ] Create `QCA_AID_assets/utils/dialog/` directory
- [ ] Create `QCA_AID_assets/utils/export/` directory
- [ ] Create `QCA_AID_assets/utils/io/` directory

### Step 1.2: Create __init__.py files

- [ ] `QCA_AID_assets/utils/__init__.py` - Main utils package
- [ ] `QCA_AID_assets/utils/llm/__init__.py` - LLM subpackage
- [ ] `QCA_AID_assets/utils/config/__init__.py` - Config subpackage
- [ ] `QCA_AID_assets/utils/tracking/__init__.py` - Tracking subpackage
- [ ] `QCA_AID_assets/utils/dialog/__init__.py` - Dialog subpackage
- [ ] `QCA_AID_assets/utils/export/__init__.py` - Export subpackage
- [ ] `QCA_AID_assets/utils/io/__init__.py` - I/O subpackage

### Step 1.3: Create common utilities file

- [ ] `QCA_AID_assets/utils/common.py` - Shared constants & helpers

### Step 1.4: Verify structure

```
QCA_AID_assets/
└── utils/
    ├── __init__.py
    ├── common.py
    ├── llm/
    │   └── __init__.py
    ├── config/
    │   └── __init__.py
    ├── tracking/
    │   └── __init__.py
    ├── dialog/
    │   └── __init__.py
    ├── export/
    │   └── __init__.py
    └── io/
        └── __init__.py
```

- [ ] Verify all directories exist
- [ ] Verify all __init__.py files created

## Phase 2-7: Migration (See REFACTORING_PLAN.md)

After Phase 1 is complete, proceed with Phase 2 (LLM System migration).

## Phase 8: Backward Compatibility Layer

- [ ] Create QCA_Utils.py as proxy that re-exports everything
- [ ] Test old imports still work: `from QCA_Utils import TokenTracker`
- [ ] Update all internal imports in QCA_AID_assets/

## Testing Checklist

### Unit Tests
- [ ] Each module has basic import tests
- [ ] No circular imports
- [ ] All exports available

### Integration Tests
- [ ] Old `from QCA_Utils import X` still works
- [ ] New `from utils.X import Y` works
- [ ] No breaking changes to existing code

### Functional Tests
- [ ] LLM API calls work
- [ ] Config loading works
- [ ] Token tracking works
- [ ] GUI dialogs appear
- [ ] Document reading works
- [ ] PDF export works

## Git Workflow

```bash
# After Phase 1:
git add QCA_AID_assets/utils/
git commit -m "refactor: phase 1 - create utils directory structure"

# After each phase:
git add QCA_AID_assets/utils/
git commit -m "refactor: phase X - migrate [component]"

# After Phase 8:
git add QCA_Utils.py
git commit -m "refactor: phase 8 - create backward compatibility layer"
```

## Success Criteria

Phase 1 is complete when:
- [ ] All directories created
- [ ] All __init__.py files exist
- [ ] Directory structure matches REFACTORING_PLAN.md
- [ ] No Python syntax errors in empty modules
- [ ] `git status` shows new directory structure

## Timeline

- **Estimated**: 1 hour for Phase 1
- **Next Phase**: Phase 2 (LLM System) - ~1.5 hours
- **Total Project**: 8 hours for all phases

## Notes

- Phase 1 has NO code migration yet, just structure
- This creates "skeleton" for Phase 2
- Can be done incrementally, one phase at a time
- Each phase is independent and testable
