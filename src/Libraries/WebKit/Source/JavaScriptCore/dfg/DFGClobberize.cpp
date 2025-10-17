/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 29, 2023.
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
#include "DFGClobberize.h"

#if ENABLE(DFG_JIT)

#include "ButterflyInlines.h"

namespace JSC { namespace DFG {

bool doesWrites(Graph& graph, Node* node)
{
    NoOpClobberize noOp;
    CheckClobberize addWrite;
    clobberize(graph, node, noOp, addWrite, noOp);
    return addWrite.result();
}

bool accessesOverlap(Graph& graph, Node* node, AbstractHeap heap)
{
    NoOpClobberize noOp;
    AbstractHeapOverlaps addAccess(heap);
    clobberize(graph, node, addAccess, addAccess, noOp);
    return addAccess.result();
}

bool writesOverlap(Graph& graph, Node* node, AbstractHeap heap)
{
    NoOpClobberize noOp;
    AbstractHeapOverlaps addWrite(heap);
    clobberize(graph, node, noOp, addWrite, noOp);
    return addWrite.result();
}

bool clobbersHeap(Graph& graph, Node* node)
{
    bool result = false;
    clobberize(
        graph, node, NoOpClobberize(),
        [&] (AbstractHeap heap) {
            switch (heap.kind()) {
            case World:
            case Heap:
                result = true;
                break;
            default:
                break;
            }
        },
        NoOpClobberize());
    return result;
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

