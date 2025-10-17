/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 8, 2025.
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
#include "DFGEdge.h"

#if ENABLE(DFG_JIT)

#include "DFGNode.h"

namespace JSC { namespace DFG {

void Edge::dump(PrintStream& out) const
{
    if (!isProved())
        out.print("Check:");
    out.print(useKind(), ":");
    if (DFG::doesKill(killStatusUnchecked()))
        out.print("Kill:");
    out.print(node());
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

