/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 12, 2023.
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
#include "AirBreakCriticalEdges.h"

#if ENABLE(B3_JIT)

#include "AirBlockInsertionSet.h"
#include "AirCode.h"

namespace JSC { namespace B3 { namespace Air {

void breakCriticalEdges(Code& code)
{
    BlockInsertionSet insertionSet(code);
    
    for (BasicBlock* block : code) {
        if (block->numSuccessors() <= 1)
            continue;
        
        for (BasicBlock*& successor : block->successorBlocks()) {
            if (successor->numPredecessors() <= 1)
                continue;
            
            BasicBlock* pad = insertionSet.insertBefore(successor, successor->frequency());
            pad->append(Jump, successor->at(0).origin);
            pad->setSuccessors(successor);
            pad->addPredecessor(block);
            successor->replacePredecessor(block, pad);
            successor = pad;
        }
    }
    
    insertionSet.execute();
}

} } } // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)

