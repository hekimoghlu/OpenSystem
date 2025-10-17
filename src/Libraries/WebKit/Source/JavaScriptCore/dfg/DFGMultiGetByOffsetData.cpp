/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 30, 2024.
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
#include "DFGMultiGetByOffsetData.h"

#if ENABLE(DFG_JIT)

#include "DFGFrozenValue.h"

namespace JSC { namespace DFG {

void GetByOffsetMethod::dumpInContext(PrintStream& out, DumpContext* context) const
{
    out.print(m_kind, ":");
    switch (m_kind) {
    case Invalid:
        out.print("<none>");
        return;
    case Constant:
        out.print(pointerDumpInContext(constant(), context));
        return;
    case Load:
        out.print(offset());
        return;
    case LoadFromPrototype:
        out.print(offset(), "@", pointerDumpInContext(prototype(), context));
        return;
    }
}

void GetByOffsetMethod::dump(PrintStream& out) const
{
    dumpInContext(out, nullptr);
}

void MultiGetByOffsetCase::dumpInContext(PrintStream& out, DumpContext* context) const
{
    out.print(inContext(m_set.toStructureSet(), context), ":", inContext(m_method, context));
}

void MultiGetByOffsetCase::dump(PrintStream& out) const
{
    dumpInContext(out, nullptr);
}

} } // namespace JSC::DFG

namespace WTF {

using namespace JSC::DFG;

void printInternal(PrintStream& out, GetByOffsetMethod::Kind kind)
{
    switch (kind) {
    case GetByOffsetMethod::Invalid:
        out.print("Invalid");
        return;
    case GetByOffsetMethod::Constant:
        out.print("Constant");
        return;
    case GetByOffsetMethod::Load:
        out.print("Load");
        return;
    case GetByOffsetMethod::LoadFromPrototype:
        out.print("LoadFromPrototype");
        return;
    }
    
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WTF

#endif // ENABLE(DFG_JIT)

