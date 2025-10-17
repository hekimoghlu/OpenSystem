/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 2, 2024.
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
#include "DeleteFromTextNodeCommand.h"

#include "CompositeEditCommand.h"
#include "Document.h"
#include "Editing.h"
#include "Text.h"

namespace WebCore {

DeleteFromTextNodeCommand::DeleteFromTextNodeCommand(Ref<Text>&& node, unsigned offset, unsigned count, EditAction editingAction)
    : SimpleEditCommand(node->document(), editingAction)
    , m_node(WTFMove(node))
    , m_offset(offset)
    , m_count(count)
{
    ASSERT(m_offset <= m_node->length());
    ASSERT(m_offset + m_count <= m_node->length());
}

void DeleteFromTextNodeCommand::doApply()
{
    auto node = protectedNode();
    if (!isEditableNode(node))
        return;

    auto result = node->substringData(m_offset, m_count);
    if (result.hasException())
        return;
    m_text = result.releaseReturnValue();
    node->deleteData(m_offset, m_count);
}

void DeleteFromTextNodeCommand::doUnapply()
{
    auto node = protectedNode();
    if (!node->hasEditableStyle())
        return;

    node->insertData(m_offset, m_text);
}

#ifndef NDEBUG
void DeleteFromTextNodeCommand::getNodesInCommand(NodeSet& nodes)
{
    addNodeAndDescendants(protectedNode().ptr(), nodes);
}
#endif

inline Ref<Text> DeleteFromTextNodeCommand::protectedNode() const
{
    return m_node;
}

} // namespace WebCore
