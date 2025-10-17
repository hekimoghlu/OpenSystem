/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 6, 2022.
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
#include "DFGFrozenValue.h"

#if ENABLE(DFG_JIT)

#include "DFGLazyJSValue.h"
#include "JSCJSValueInlines.h"

namespace JSC { namespace DFG {

FrozenValue* FrozenValue::emptySingleton()
{
    static FrozenValue empty;
    return &empty;
}

String FrozenValue::tryGetString(Graph& graph)
{
    return LazyJSValue(this).tryGetString(graph);
}

void FrozenValue::dumpInContext(PrintStream& out, DumpContext* context) const
{
    if (!!m_value && m_value.isCell())
        out.print(m_strength, ":");
    m_value.dumpInContextAssumingStructure(out, context, m_structure);
}

void FrozenValue::dump(PrintStream& out) const
{
    dumpInContext(out, nullptr);
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
