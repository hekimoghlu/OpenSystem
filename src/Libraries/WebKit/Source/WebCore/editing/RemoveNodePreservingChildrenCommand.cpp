/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 3, 2023.
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
#include "RemoveNodePreservingChildrenCommand.h"

#include "Editing.h"
#include "Node.h"
#include <wtf/Assertions.h>

namespace WebCore {

RemoveNodePreservingChildrenCommand::RemoveNodePreservingChildrenCommand(Ref<Node>&& node, ShouldAssumeContentIsAlwaysEditable shouldAssumeContentIsAlwaysEditable, EditAction editingAction)
    : CompositeEditCommand(node->document(), editingAction)
    , m_node(WTFMove(node))
    , m_shouldAssumeContentIsAlwaysEditable(shouldAssumeContentIsAlwaysEditable)
{
}

void RemoveNodePreservingChildrenCommand::doApply()
{
    Vector<Ref<Node>> children;
    auto node = protectedNode();
    RefPtr parent { node->parentNode() };
    if (!parent || (m_shouldAssumeContentIsAlwaysEditable == DoNotAssumeContentIsAlwaysEditable && !isEditableNode(*parent)))
        return;

    for (Node* child = node->firstChild(); child; child = child->nextSibling())
        children.append(*child);

    size_t size = children.size();
    for (size_t i = 0; i < size; ++i) {
        auto child = WTFMove(children[i]);
        removeNode(child, m_shouldAssumeContentIsAlwaysEditable);
        insertNodeBefore(WTFMove(child), node, m_shouldAssumeContentIsAlwaysEditable);
    }
    removeNode(node, m_shouldAssumeContentIsAlwaysEditable);
}

}
