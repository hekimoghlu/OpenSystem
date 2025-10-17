/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 16, 2023.
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
#include "DFGClobbersExitState.h"

#if ENABLE(DFG_JIT)

#include "ButterflyInlines.h"
#include "DFGClobberize.h"
#include "DFGNode.h"

namespace JSC { namespace DFG {

bool clobbersExitState(Graph& graph, Node* node)
{
    // There are certain nodes whose effect on the exit state has nothing to do with what they
    // normally clobber.
    switch (node->op()) {
    case InitializeEntrypointArguments:
    case MovHint:
    case ZombieHint:
    case PutHint:
    case KillStack:
        return true;

    case SetLocal:
    case PutStack:
        // These nodes write to the stack, but they may only do so after we have already had a MovHint
        // for the exact same value and the same stack location. Hence, they have no further effect on
        // exit state.
        return false;

    case ArrayifyToStructure:
    case Arrayify:
    case NewObject:
    case NewGenerator:
    case NewAsyncGenerator:
    case NewInternalFieldObject:
    case NewRegexp:
    case NewMap:
    case NewSet:
    case NewStringObject:
    case NewBoundFunction:
    case PhantomNewObject:
    case MaterializeNewObject:
    case PhantomNewFunction:
    case PhantomNewGeneratorFunction:
    case PhantomNewAsyncGeneratorFunction:
    case PhantomNewAsyncFunction:
    case PhantomNewInternalFieldObject:
    case MaterializeNewInternalFieldObject:
    case PhantomCreateActivation:
    case MaterializeCreateActivation:
    case PhantomNewRegexp:
    case CountExecution:
    case SuperSamplerBegin:
    case SuperSamplerEnd:
    case StoreBarrier:
    case FencedStoreBarrier:
    case AllocatePropertyStorage:
    case ReallocatePropertyStorage:
    case FilterCallLinkStatus:
    case FilterGetByStatus:
    case FilterPutByStatus:
    case FilterInByStatus:
    case FilterDeleteByStatus:
    case FilterCheckPrivateBrandStatus:
    case FilterSetPrivateBrandStatus:
    case TryGetById:
        // These do clobber memory, but nothing that is observable. It may be nice to separate the
        // heaps into those that are observable and those that aren't, but we don't do that right now.
        // FIXME: https://bugs.webkit.org/show_bug.cgi?id=148440
        return false;

    case CreateActivation:
        // Like above, but with the activation allocation caveat.
        return node->castOperand<SymbolTable*>()->singleton().isStillValid();

    case NewFunction:
    case NewGeneratorFunction:
    case NewAsyncGeneratorFunction:
    case NewAsyncFunction:
        // Like above, but with the JSFunction allocation caveat.
        return node->castOperand<FunctionExecutable*>()->singleton().isStillValid();

    default:
        // For all other nodes, we just care about whether they write to something other than SideState.
        bool result = false;
        clobberize(
            graph, node, NoOpClobberize(),
            [&] (const AbstractHeap& heap) {
                // There shouldn't be such a thing as a strict subtype of SideState or HeapObjectCount.
                // That's what allows us to use a fast != check, below.
                ASSERT(!heap.isStrictSubtypeOf(SideState) && !heap.isStrictSubtypeOf(HeapObjectCount));

                if (heap != SideState && heap != HeapObjectCount)
                    result = true;
            },
            NoOpClobberize());
        return result;
    }
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
