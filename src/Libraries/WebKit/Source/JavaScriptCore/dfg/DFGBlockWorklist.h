/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 12, 2025.
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

#include "DFGBasicBlock.h"
#include "DFGBlockSet.h"
#include <wtf/GraphNodeWorklist.h>

namespace JSC { namespace DFG {

class BasicBlock;

typedef GraphNodeWorklist<BasicBlock*, BlockSet> BlockWorklist;

// When you say BlockWith<int> you should read it as "block with an int".
template<typename T> using BlockWith = GraphNodeWith<BasicBlock*, T>;

// Extended block worklist is useful for enqueueing some meta-data along with the block. It also
// permits forcibly enqueueing things even if the block has already been seen. It's useful for
// things like building a spanning tree, in which case T (the auxiliary payload) would be the
// successor index.
template<typename T> using ExtendedBlockWorklist = ExtendedGraphNodeWorklist<BasicBlock*, T, BlockSet>;

typedef GraphVisitOrder VisitOrder;

typedef GraphNodeWithOrder<BasicBlock*> BlockWithOrder;

typedef PostOrderGraphNodeWorklist<BasicBlock*, BlockSet> PostOrderBlockWorklist;

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)
