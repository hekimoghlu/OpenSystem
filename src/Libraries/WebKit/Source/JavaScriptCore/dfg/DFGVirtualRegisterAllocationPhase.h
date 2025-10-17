/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 4, 2023.
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

#include "DFGPhase.h"

namespace JSC { namespace DFG {

class Graph;

// Prior to running this phase, we have no idea where in the call frame nodes
// will have their values spilled. This phase fixes that by giving each node
// a spill slot. The spill slot index (i.e. the virtual register) is also used
// for look-up tables for the linear scan register allocator that the backend
// uses.

bool performVirtualRegisterAllocation(Graph&);

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
