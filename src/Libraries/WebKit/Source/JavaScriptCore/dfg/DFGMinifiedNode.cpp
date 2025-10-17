/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 28, 2024.
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
#include "DFGMinifiedNode.h"

#if ENABLE(DFG_JIT)

#include "DFGMinifiedIDInlines.h"
#include "DFGNode.h"
#include "JSCJSValueInlines.h"

namespace JSC { namespace DFG {

MinifiedNode MinifiedNode::fromNode(Node* node)
{
    ASSERT(belongsInMinifiedGraph(node->op()));
    MinifiedNode result;
    result.m_id = MinifiedID(node);
    result.m_hasConstant = hasConstant(node->op());
    result.m_isPhantomDirectArguments = node->op() == PhantomDirectArguments;
    result.m_isPhantomClonedArguments = node->op() == PhantomClonedArguments;
    if (hasConstant(node->op()))
        result.m_info = JSValue::encode(node->asJSValue());
    else {
        ASSERT(node->op() == PhantomDirectArguments || node->op() == PhantomClonedArguments);
        result.m_info = std::bit_cast<uintptr_t>(node->origin.semantic.inlineCallFrame());
    }
    return result;
}

} } // namespace JSC::DFG

#endif // ENABLE(DFG_JIT)

