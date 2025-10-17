/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 31, 2024.
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
#include "InsertNodeBeforeCommand.h"

#include "CompositeEditCommand.h"
#include "Document.h"
#include "Editing.h"
#include "RenderElement.h"
#include "Text.h"

namespace WebCore {

InsertNodeBeforeCommand::InsertNodeBeforeCommand(Ref<Node>&& insertChild, Node& refChild, ShouldAssumeContentIsAlwaysEditable shouldAssumeContentIsAlwaysEditable, EditAction editingAction)
    : SimpleEditCommand(refChild.document(), editingAction)
    , m_insertChild(WTFMove(insertChild))
    , m_refChild(refChild)
    , m_shouldAssumeContentIsAlwaysEditable(shouldAssumeContentIsAlwaysEditable)
{
    ASSERT(!m_insertChild->parentNode());
    ASSERT(m_refChild->parentNode());

    ASSERT(m_refChild->parentNode()->hasEditableStyle() || !m_refChild->parentNode()->renderer());
}

void InsertNodeBeforeCommand::doApply()
{
    RefPtr parent = m_refChild->parentNode();
    if (!parent || (m_shouldAssumeContentIsAlwaysEditable == DoNotAssumeContentIsAlwaysEditable && !isEditableNode(*parent)))
        return;
    ASSERT(isEditableNode(*parent));

    parent->insertBefore(protectedInsertChild(), m_refChild.copyRef());
}

void InsertNodeBeforeCommand::doUnapply()
{
    auto insertChild = protectedInsertChild();
    if (!isEditableNode(insertChild))
        return;

    insertChild->remove();
}

#ifndef NDEBUG
void InsertNodeBeforeCommand::getNodesInCommand(NodeSet& nodes)
{
    addNodeAndDescendants(m_insertChild.ptr(), nodes);
    addNodeAndDescendants(m_refChild.ptr(), nodes);
}
#endif

}
