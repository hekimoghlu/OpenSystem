/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 24, 2025.
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
#include "AppendNodeCommand.h"

#include "AXObjectCache.h"
#include "CompositeEditCommand.h"
#include "Document.h"
#include "Editing.h"
#include "RenderElement.h"
#include "Text.h"

namespace WebCore {

AppendNodeCommand::AppendNodeCommand(Ref<ContainerNode>&& parent, Ref<Node>&& node, EditAction editingAction)
    : SimpleEditCommand(parent->document(), editingAction)
    , m_parent(WTFMove(parent))
    , m_node(WTFMove(node))
{
    ASSERT(!m_node->parentNode());
}

void AppendNodeCommand::doApply()
{
    auto parent = protectedParent();
    if (!parent->hasEditableStyle() && parent->renderer())
        return;

    parent->appendChild(m_node);
}

void AppendNodeCommand::doUnapply()
{
    if (!m_parent->hasEditableStyle())
        return;

    protectedNode()->remove();
}

#ifndef NDEBUG
void AppendNodeCommand::getNodesInCommand(NodeSet& nodes)
{
    addNodeAndDescendants(m_parent.ptr(), nodes);
    addNodeAndDescendants(m_node.ptr(), nodes);
}
#endif

} // namespace WebCore
