/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 4, 2022.
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
#include "DFGHeapLocation.h"

#if ENABLE(DFG_JIT)

#include "JSCJSValueInlines.h"

namespace JSC { namespace DFG {

void HeapLocation::dump(PrintStream& out) const
{
    out.print(m_kind, ":", m_heap);
    
    if (!m_base)
        return;
    
    out.print("[", m_base);
    if (m_index)
        out.print(", ", m_index);
    out.print("]");
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

