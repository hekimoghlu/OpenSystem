/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 6, 2021.
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
#include "JSCast.h"
#include "MarkedBlock.h"
#include "PreciseAllocation.h"
#include "VM.h"

namespace JSC {

inline VM& CellContainer::vm() const
{
    if (isPreciseAllocation())
        return preciseAllocation().vm();
    return markedBlock().vm();
}

inline JSC::Heap* CellContainer::heap() const
{
    return &vm().heap;
}

inline bool CellContainer::isMarked(HeapCell* cell) const
{
    if (isPreciseAllocation())
        return preciseAllocation().isMarked();
    return markedBlock().isMarked(cell);
}

inline bool CellContainer::isMarked(HeapVersion markingVersion, HeapCell* cell) const
{
    if (isPreciseAllocation())
        return preciseAllocation().isMarked();
    return markedBlock().isMarked(markingVersion, cell);
}

inline void CellContainer::noteMarked()
{
    if (!isPreciseAllocation())
        markedBlock().noteMarked();
}

inline void CellContainer::assertValidCell(VM& vm, HeapCell* cell) const
{
    if (isPreciseAllocation())
        preciseAllocation().assertValidCell(vm, cell);
    else
        markedBlock().assertValidCell(vm, cell);
}

inline size_t CellContainer::cellSize() const
{
    if (isPreciseAllocation())
        return preciseAllocation().cellSize();
    return markedBlock().cellSize();
}

inline WeakSet& CellContainer::weakSet() const
{
    if (isPreciseAllocation())
        return preciseAllocation().weakSet();
    return markedBlock().weakSet();
}

inline bool CellContainer::areMarksStale() const
{
    if (isPreciseAllocation())
        return false;
    return markedBlock().areMarksStale();
}

} // namespace JSC

