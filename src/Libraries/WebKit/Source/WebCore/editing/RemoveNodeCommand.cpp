/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 10, 2024.
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
#include "RemoveNodeCommand.h"

#include "CompositeEditCommand.h"
#include "Editing.h"
#include "RenderElement.h"
#include <wtf/Assertions.h>

namespace WebCore {

RemoveNodeCommand::RemoveNodeCommand(Ref<Node>&& node, ShouldAssumeContentIsAlwaysEditable shouldAssumeContentIsAlwaysEditable, EditAction editingAction)
    : SimpleEditCommand(node->document(), editingAction)
    , m_node(WTFMove(node))
    , m_shouldAssumeContentIsAlwaysEditable(shouldAssumeContentIsAlwaysEditable)
{
    ASSERT(m_node->parentNode());
}

void RemoveNodeCommand::doApply()
{
    auto node = protectedNode();
    RefPtr parent = node->parentNode();
    if (!parent || (m_shouldAssumeContentIsAlwaysEditable == DoNotAssumeContentIsAlwaysEditable
        && !isEditableNode(*parent) && parent->renderer()))
        return;
    ASSERT(isEditableNode(*parent) || !parent->renderer());

    m_parent = WTFMove(parent);
    m_refChild = node->nextSibling();

    node->remove();
}

void RemoveNodeCommand::doUnapply()
{
    RefPtr<ContainerNode> parent = WTFMove(m_parent);
    RefPtr<Node> refChild = WTFMove(m_refChild);
    if (!parent || !parent->hasEditableStyle())
        return;

    parent->insertBefore(protectedNode(), WTFMove(refChild));
}

#ifndef NDEBUG
void RemoveNodeCommand::getNodesInCommand(NodeSet& nodes)
{
    addNodeAndDescendants(m_parent.get(), nodes);
    addNodeAndDescendants(m_refChild.get(), nodes);
    addNodeAndDescendants(m_node.ptr(), nodes);
}
#endif

}
