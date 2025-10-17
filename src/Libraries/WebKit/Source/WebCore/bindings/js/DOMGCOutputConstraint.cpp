/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 11, 2024.
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
#include "DOMGCOutputConstraint.h"

#include "WebCoreJSClientData.h"
#include <JavaScriptCore/BlockDirectoryInlines.h>
#include <JavaScriptCore/HeapInlines.h>
#include <JavaScriptCore/JSCellInlines.h>
#include <JavaScriptCore/MarkedBlockInlines.h>
#include <JavaScriptCore/SlotVisitorInlines.h>
#include <JavaScriptCore/SubspaceInlines.h>
#include <JavaScriptCore/VM.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

using namespace JSC;

WTF_MAKE_TZONE_ALLOCATED_IMPL(DOMGCOutputConstraint);

DOMGCOutputConstraint::DOMGCOutputConstraint(VM& vm, JSHeapData& heapData)
    : MarkingConstraint("Domo", "DOM Output", ConstraintVolatility::SeldomGreyed, ConstraintConcurrency::Concurrent, ConstraintParallelism::Parallel)
    , m_vm(vm)
    , m_heapData(heapData)
    , m_lastExecutionVersion(vm.heap.mutatorExecutionVersion())
{
}

DOMGCOutputConstraint::~DOMGCOutputConstraint()
{
}

template<typename Visitor>
void DOMGCOutputConstraint::executeImplImpl(Visitor& visitor)
{
    Heap& heap = m_vm.heap;
    
    if (heap.mutatorExecutionVersion() == m_lastExecutionVersion)
        return;
    
    m_lastExecutionVersion = heap.mutatorExecutionVersion();
    
    m_heapData.forEachOutputConstraintSpace(
        [&] (Subspace& subspace) {
            auto func = [] (Visitor& visitor, HeapCell* heapCell, HeapCell::Kind) {
                SetRootMarkReasonScope rootScope(visitor, RootMarkReason::DOMGCOutput);
                JSCell* cell = static_cast<JSCell*>(heapCell);
                cell->methodTable()->visitOutputConstraints(cell, visitor);
            };
            
            RefPtr<SharedTask<void(Visitor&)>> task = subspace.template forEachMarkedCellInParallel<Visitor>(func);
            visitor.addParallelConstraintTask(task);
        });
}
        
void DOMGCOutputConstraint::executeImpl(AbstractSlotVisitor& visitor) { executeImplImpl(visitor); }
void DOMGCOutputConstraint::executeImpl(SlotVisitor& visitor) { executeImplImpl(visitor); }

} // namespace WebCore

