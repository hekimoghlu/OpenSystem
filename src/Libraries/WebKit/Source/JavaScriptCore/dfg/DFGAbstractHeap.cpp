/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 12, 2023.
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
#include "DFGAbstractHeap.h"

#if ENABLE(DFG_JIT)

namespace JSC { namespace DFG {

void AbstractHeap::Payload::dump(PrintStream& out) const
{
    if (isTop())
        out.print("TOP");
    else
        out.print(value());
}

void AbstractHeap::Payload::dumpAsOperand(PrintStream& out) const
{
    if (isTop())
        out.print("TOP");
    else
        out.print(Operand::fromBits(value()));
}

void AbstractHeap::dump(PrintStream& out) const
{
    out.print(kind());
    if (kind() == InvalidAbstractHeap || kind() == World || kind() == Heap || payload().isTop())
        return;
    if (kind() == DOMState) {
        out.print("(", DOMJIT::HeapRange::fromRaw(payload().value32()), ")");
        return;
    }
    if (kind() == Stack) {
        out.print("(");
        payload().dumpAsOperand(out);
        out.print(")");
        return;
    }

    out.print("(", payload(), ")");
}

} } // namespace JSC::DFG

namespace WTF {

using namespace JSC::DFG;

void printInternal(PrintStream& out, AbstractHeapKind kind)
{
    switch (kind) {
#define ABSTRACT_HEAP_DUMP(name)            \
    case AbstractHeapKind::name:            \
        out.print(#name);                   \
        return;
    FOR_EACH_ABSTRACT_HEAP_KIND(ABSTRACT_HEAP_DUMP)
#undef ABSTRACT_HEAP_DUMP
    }
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WTF

#endif // ENABLE(DFG_JIT)

