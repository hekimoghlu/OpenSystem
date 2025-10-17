/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 19, 2024.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#pragma once

#include "CellList.h"
#include "Heap.h"
#include <wtf/MonotonicTime.h>
#include <wtf/ScopedLambda.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueArray.h>

namespace JSC {

class JSCell;
class MarkedBlock;

class HeapVerifier {
    WTF_MAKE_TZONE_ALLOCATED(HeapVerifier);
public:
    enum class Phase {
        BeforeGC,
        BeforeMarking,
        AfterMarking,
        AfterGC
    };

    HeapVerifier(Heap*, unsigned numberOfGCCyclesToRecord);

    void startGC();
    void endGC();

    void gatherLiveCells(Phase);
    void trimDeadCells();
    void verify(Phase);

    static const char* phaseName(Phase);
    
    // Scans all previously recorded CellLists and checks if the specified
    // cell was in any of those lists.
    JS_EXPORT_PRIVATE static void checkIfRecorded(uintptr_t maybeCell);

    // Returns false if anything is found to be inconsistent/incorrect about the specified cell.
    JS_EXPORT_PRIVATE static bool validateCell(HeapCell*, VM* expectedVM = nullptr);

private:
    struct GCCycle {
        WTF_MAKE_STRUCT_FAST_ALLOCATED;

        GCCycle()
            : before("Before Marking")
            , after("After Marking")
        {
        }

        void reset()
        {
            before.reset();
            after.reset();
        }

        CollectionScope scope;
        MonotonicTime timestamp;
        CellList before;
        CellList after;
    };

    void incrementCycle() { m_currentCycle = (m_currentCycle + 1) % m_numberOfCycles; }
    GCCycle& currentCycle() { return m_cycles[m_currentCycle]; }
    GCCycle& cycleForIndex(int cycleIndex)
    {
        ASSERT(cycleIndex <= 0 && cycleIndex > -m_numberOfCycles);
        cycleIndex += m_currentCycle;
        if (cycleIndex < 0)
            cycleIndex += m_numberOfCycles;
        ASSERT(cycleIndex < m_numberOfCycles);
        return m_cycles[cycleIndex];
    }

    CellList* cellListForGathering(Phase);
    bool verifyCellList(Phase, CellList&);
    static bool validateJSCell(VM* expectedVM, JSCell*, CellProfile*, CellList*, const ScopedLambda<void()>& printHeaderIfNeeded, const char* prefix = "");

    void printVerificationHeader();

    void checkIfRecorded(HeapCell* maybeHeapCell);
    void reportCell(CellProfile&, int cycleIndex, HeapVerifier::GCCycle&, CellList&, const char* prefix = nullptr);

    Heap* m_heap;
    int m_currentCycle;
    int m_numberOfCycles;
    bool m_didPrintLogs { false };
    UniqueArray<GCCycle> m_cycles;
};

} // namespace JSC
