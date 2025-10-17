/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 28, 2021.
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
#include "JSDestructibleObjectHeapCellType.h"

#include "JSCJSValueInlines.h"
#include "JSDestructibleObject.h"
#include "MarkedBlockInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace JSC {

WTF_MAKE_TZONE_ALLOCATED_IMPL(JSDestructibleObjectHeapCellType);

struct JSDestructibleObjectDestroyFunc {
    WTF_FORBID_HEAP_ALLOCATION;
public:
    ALWAYS_INLINE void operator()(VM&, JSCell* cell) const
    {
        static_cast<JSDestructibleObject*>(cell)->classInfo()->methodTable.destroy(cell);
    }
};

JSDestructibleObjectHeapCellType::JSDestructibleObjectHeapCellType()
    : HeapCellType(CellAttributes(NeedsDestruction, HeapCell::JSCell))
{
}

JSDestructibleObjectHeapCellType::~JSDestructibleObjectHeapCellType() = default;

void JSDestructibleObjectHeapCellType::finishSweep(MarkedBlock::Handle& handle, FreeList* freeList) const
{
    handle.finishSweepKnowingHeapCellType(freeList, JSDestructibleObjectDestroyFunc());
}

void JSDestructibleObjectHeapCellType::destroy(VM& vm, JSCell* cell) const
{
    JSDestructibleObjectDestroyFunc()(vm, cell);
}

} // namespace JSC

