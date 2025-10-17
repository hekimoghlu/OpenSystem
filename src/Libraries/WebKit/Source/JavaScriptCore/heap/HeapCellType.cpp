/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 28, 2023.
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
#include "HeapCellType.h"

#include "JSCInlines.h"
#include "MarkedBlockInlines.h"

namespace JSC {

// Writing it this way ensures that when you pass this as a functor, the callee is specialized for
// this callback. If you wrote this as a normal function then the callee would be specialized for
// the function's type and it would have indirect calls to that function. And unlike a lambda, it's
// possible to mark this ALWAYS_INLINE.
struct DefaultDestroyFunc {
    ALWAYS_INLINE void operator()(VM&, JSCell* cell) const
    {
        ASSERT(cell->structureID());
        Structure* structure = cell->structure();
        ASSERT(structure->typeInfo().structureIsImmortal());
        const ClassInfo* classInfo = structure->classInfoForCells();
        MethodTable::DestroyFunctionPtr destroy = classInfo->methodTable.destroy;
        destroy(cell);
    }
};

HeapCellType::HeapCellType(CellAttributes attributes)
    : m_attributes(attributes)
{
}

HeapCellType::~HeapCellType() = default;

void HeapCellType::finishSweep(MarkedBlock::Handle& block, FreeList* freeList) const
{
    block.finishSweepKnowingHeapCellType(freeList, DefaultDestroyFunc());
}

void HeapCellType::destroy(VM& vm, JSCell* cell) const
{
    DefaultDestroyFunc()(vm, cell);
}

} // namespace JSC

