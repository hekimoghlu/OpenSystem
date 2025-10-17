/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 18, 2024.
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
#include "HeapCell.h"

#include "HeapCellInlines.h"
#include "MarkedBlockInlines.h"
#include <wtf/PrintStream.h>

namespace JSC {

// isPendingDestruction must not be called on a freed (i.e. already destructed) HeapCell
// since the backing MarkedBlock may have been freed too.
bool HeapCell::isPendingDestruction()
{
    if (isPreciseAllocation())
        return !preciseAllocation().isLive();
    auto& markedBlockHandle = markedBlock().handle();
    // If the block is freelisted then either:
    // (1) The cell is not on the FreeList, in which case it is newly allocated or
    // (2) The cell is on the FreeList, in which case it is free.
    // In either case, the destructor is not pending.
    // (And as indicated above, it's not legal to call this method in state #2).
    if (markedBlockHandle.isFreeListed())
        return false;
    return !markedBlockHandle.isLive(this);
}

} // namespace JSC

namespace WTF {

using namespace JSC;

void printInternal(PrintStream& out, HeapCell::Kind kind)
{
    switch (kind) {
    case HeapCell::JSCell:
        out.print("JSCell");
        return;
    case HeapCell::JSCellWithIndexingHeader:
        out.print("JSCellWithIndexingHeader");
        return;
    case HeapCell::Auxiliary:
        out.print("Auxiliary");
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WTF

