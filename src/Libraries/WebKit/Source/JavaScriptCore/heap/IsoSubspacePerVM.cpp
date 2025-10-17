/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 27, 2024.
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
#include "IsoSubspacePerVM.h"

#include "HeapInlines.h"
#include "MarkedSpaceInlines.h"

namespace JSC {

IsoSubspacePerVM::IsoSubspacePerVM(Function<SubspaceParameters(Heap&)> subspaceParameters)
    : m_subspaceParameters(WTFMove(subspaceParameters))
{
}

IsoSubspacePerVM::~IsoSubspacePerVM()
{
    UNREACHABLE_FOR_PLATFORM();
}

IsoSubspace& IsoSubspacePerVM::isoSubspaceforHeap(Locker<Lock>&, JSC::Heap& heap)
{
    auto result = m_subspacePerHeap.add(&heap, nullptr);
    if (result.isNewEntry) {
        SubspaceParameters params = m_subspaceParameters(heap);
        constexpr bool usePreciseAllocationsOnly = false;
        constexpr uint8_t numberOfLowerTierPreciseCells = 0;
        result.iterator->value = new IsoSubspace(params.name, heap, *params.heapCellType, params.size, usePreciseAllocationsOnly, numberOfLowerTierPreciseCells);

        Locker locker { heap.lock() };
        heap.perVMIsoSubspaces.append(this);
    }
    return *result.iterator->value;
}

GCClient::IsoSubspace& IsoSubspacePerVM::clientIsoSubspaceforVM(VM& vm)
{
    Locker locker { m_lock };
    auto result = m_clientSubspacePerVM.add(&vm, nullptr);
    if (!result.isNewEntry && result.iterator->value)
        return *result.iterator->value;

    IsoSubspace& subspace = isoSubspaceforHeap(locker, vm.heap);

    result.iterator->value = new GCClient::IsoSubspace(subspace);
    vm.clientHeap.perVMIsoSubspaces.append(this);
    return *result.iterator->value;
}

void IsoSubspacePerVM::releaseIsoSubspace(JSC::Heap& heap)
{
    IsoSubspace* subspace;
    {
        Locker locker { m_lock };
        subspace = m_subspacePerHeap.take(&heap);
    }
    delete subspace;
}

void IsoSubspacePerVM::releaseClientIsoSubspace(VM& vm)
{
    GCClient::IsoSubspace* clientSubspace;
    {
        Locker locker { m_lock };
        clientSubspace = m_clientSubspacePerVM.take(&vm);
    }
    delete clientSubspace;
}

} // namespace JSC

