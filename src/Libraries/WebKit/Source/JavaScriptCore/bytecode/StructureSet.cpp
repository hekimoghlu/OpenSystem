/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 21, 2022.
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
#include "StructureSet.h"

#include "HeapInlines.h"
#include <wtf/CommaPrinter.h>

namespace JSC {

template<typename Visitor>
void StructureSet::markIfCheap(Visitor& visitor) const
{
    for (Structure* structure : *this)
        structure->markIfCheap(visitor);
}

template void StructureSet::markIfCheap(AbstractSlotVisitor&) const;
template void StructureSet::markIfCheap(SlotVisitor&) const;

bool StructureSet::isStillAlive(VM& vm) const
{
    for (Structure* structure : *this) {
        if (!vm.heap.isMarked(structure))
            return false;
    }
    return true;
}

void StructureSet::dumpInContext(PrintStream& out, DumpContext* context) const
{
    CommaPrinter comma;
    out.print("["_s);
    forEach([&] (Structure* structure) { out.print(comma, inContext(*structure, context)); });
    out.print("]"_s);
}

void StructureSet::dump(PrintStream& out) const
{
    dumpInContext(out, nullptr);
}

} // namespace JSC

