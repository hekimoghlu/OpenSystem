/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 9, 2022.
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
#include "HandleSet.h"

#include "HandleBlock.h"
#include "HandleBlockInlines.h"
#include "JSCJSValueInlines.h"

namespace JSC {

HandleSet::HandleSet(VM& vm)
    : m_vm(vm)
{
    grow();
}

HandleSet::~HandleSet()
{
    while (!m_blockList.isEmpty())
        HandleBlock::destroy(m_blockList.removeHead());
}

void HandleSet::grow()
{
    HandleBlock* newBlock = HandleBlock::create(this);
    m_blockList.append(newBlock);

    for (int i = newBlock->nodeCapacity() - 1; i >= 0; --i) {
        Node* node = newBlock->nodeAtIndex(i);
        new (NotNull, node) Node;
        m_freeList.push(node);
    }
}

template<typename Visitor>
void HandleSet::visitStrongHandles(Visitor& visitor)
{
    for (Node& node : m_strongList) {
#if ENABLE(GC_VALIDATION)
        RELEASE_ASSERT(isLiveNode(&node));
#endif
        visitor.appendUnbarriered(*node.slot());
    }
}

template void HandleSet::visitStrongHandles(AbstractSlotVisitor&);
template void HandleSet::visitStrongHandles(SlotVisitor&);

unsigned HandleSet::protectedGlobalObjectCount()
{
    unsigned count = 0;
    for (Node& node : m_strongList) {
        JSValue value = *node.slot();
        if (value.isObject() && asObject(value.asCell())->isGlobalObject())
            count++;
    }
    return count;
}

#if ENABLE(GC_VALIDATION) || ASSERT_ENABLED
bool HandleSet::isLiveNode(Node* node)
{
    if (node->prev()->next() != node)
        return false;
    if (node->next()->prev() != node)
        return false;
        
    return true;
}
#endif // ENABLE(GC_VALIDATION) || ASSERT_ENABLED

} // namespace JSC
