/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 9, 2025.
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

#include "CellContainer.h"
#include "HeapCell.h"
#include "PreciseAllocation.h"
#include "VM.h"

namespace JSC {

ALWAYS_INLINE bool HeapCell::isPreciseAllocation() const
{
    return PreciseAllocation::isPreciseAllocation(const_cast<HeapCell*>(this));
}

ALWAYS_INLINE CellContainer HeapCell::cellContainer() const
{
    if (isPreciseAllocation())
        return preciseAllocation();
    return markedBlock();
}

ALWAYS_INLINE MarkedBlock& HeapCell::markedBlock() const
{
    return *MarkedBlock::blockFor(this);
}

ALWAYS_INLINE PreciseAllocation& HeapCell::preciseAllocation() const
{
    return *PreciseAllocation::fromCell(const_cast<HeapCell*>(this));
}

ALWAYS_INLINE JSC::Heap* HeapCell::heap() const
{
    return &vm().heap;
}

ALWAYS_INLINE VM& HeapCell::vm() const
{
    if (isPreciseAllocation())
        return preciseAllocation().vm();
    return markedBlock().vm();
}
    
ALWAYS_INLINE size_t HeapCell::cellSize() const
{
    if (isPreciseAllocation())
        return preciseAllocation().cellSize();
    return markedBlock().cellSize();
}

ALWAYS_INLINE CellAttributes HeapCell::cellAttributes() const
{
    if (isPreciseAllocation())
        return preciseAllocation().attributes();
    return markedBlock().attributes();
}

ALWAYS_INLINE DestructionMode HeapCell::destructionMode() const
{
    return cellAttributes().destruction;
}

ALWAYS_INLINE HeapCell::Kind HeapCell::cellKind() const
{
    return cellAttributes().cellKind;
}

ALWAYS_INLINE Subspace* HeapCell::subspace() const
{
    if (isPreciseAllocation())
        return preciseAllocation().subspace();
    return markedBlock().subspace();
}

ALWAYS_INLINE void HeapCell::notifyNeedsDestruction() const
{
    ASSERT(!isPreciseAllocation());
    ASSERT(destructionMode() == MayNeedDestruction);
    markedBlock().handle().setIsDestructible(true);
}

} // namespace JSC

