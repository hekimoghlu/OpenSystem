/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 11, 2025.
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

#if ENABLE(DFG_JIT)

#include "DFGCommon.h"

namespace JSC { namespace DFG {

class Graph;

// Block-local common subexpression elimination. It uses clobberize() for heap
// modeling, which is quite precise. This phase is known to produce big wins on
// a few benchmarks, and is relatively cheap to run.
//
// Note that this phase also gets rid of Identity nodes, which means that it's
// currently not an optional phase. Basically, DFG IR doesn't have use-lists,
// so there is no instantaneous replaceAllUsesWith operation. Instead, you turn
// a node into an Identity and wait for CSE to clean it up.
bool performLocalCSE(Graph&);

// Same, but global. Only works for SSA. This will find common subexpressions
// both in the same block and in any block that dominates the current block. It
// has no limits on how far it will look for load-elimination opportunities.
bool performGlobalCSE(Graph&);

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
