/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 3, 2024.
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
#include "config.h"
#include "B3LegalizeMemoryOffsets.h"

#if ENABLE(B3_JIT)

#include "B3InsertionSet.h"
#include "B3MemoryValueInlines.h"
#include "B3PhaseScope.h"
#include "B3ProcedureInlines.h"
#include "B3ValueInlines.h"

namespace JSC { namespace B3 {

namespace {

class LegalizeMemoryOffsets {
public:
    LegalizeMemoryOffsets(Procedure& proc)
        : m_proc(proc)
        , m_insertionSet(proc)
    {
    }

    void run()
    {
        // FIXME: Perhaps this should be moved to lowerMacros, and quirks mode can impose the requirement
        // that the offset is legal. But for now this is sort of OK because we run pureCSE after. Also,
        // we should probably have something better than just pureCSE to clean up the code that this
        // introduces.
        // https://bugs.webkit.org/show_bug.cgi?id=169246
        
        for (BasicBlock* block : m_proc) {
            for (unsigned index = 0; index < block->size(); ++index) {
                MemoryValue* memoryValue = block->at(index)->as<MemoryValue>();
                if (!memoryValue)
                    continue;
                
                if (!memoryValue->isLegalOffset(memoryValue->offset())) {
                    Value* base = memoryValue->lastChild();
                    Value* offsetValue = m_insertionSet.insertIntConstant(index, memoryValue->origin(), pointerType(), memoryValue->offset());
                    Value* resolvedAddress = m_proc.add<Value>(Add, memoryValue->origin(), base, offsetValue);
                    m_insertionSet.insertValue(index, resolvedAddress);

                    memoryValue->lastChild() = resolvedAddress;
                    memoryValue->setOffset(0);
                }
            }
            m_insertionSet.execute(block);
        }
    }

    Procedure& m_proc;
    InsertionSet m_insertionSet;
};

} // anonymous namespace

void legalizeMemoryOffsets(Procedure& proc)
{
    PhaseScope phaseScope(proc, "legalizeMemoryOffsets"_s);
    LegalizeMemoryOffsets legalizeMemoryOffsets(proc);
    legalizeMemoryOffsets.run();
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)

