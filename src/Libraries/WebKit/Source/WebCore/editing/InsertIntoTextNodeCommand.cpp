/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 15, 2022.
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
#include "InsertIntoTextNodeCommand.h"

#include "CompositeEditCommand.h"
#include "Document.h"
#include "Editor.h"
#include "EditorClient.h"
#include "LocalFrame.h"
#include "RenderText.h"
#include "Settings.h"
#include "Text.h"

#if PLATFORM(IOS_FAMILY)
#include "RenderText.h"
#endif

namespace WebCore {

InsertIntoTextNodeCommand::InsertIntoTextNodeCommand(Ref<Text>&& node, unsigned offset, const String& text, EditAction editingAction)
    : SimpleEditCommand(node->document(), editingAction)
    , m_node(WTFMove(node))
    , m_offset(offset)
    , m_text(text)
{
    ASSERT(m_offset <= m_node->length());
    ASSERT(!m_text.isEmpty());
}

void InsertIntoTextNodeCommand::doApply()
{
    bool passwordEchoEnabled = document().settings().passwordEchoEnabled() && !document().editor().client()->shouldSuppressPasswordEcho();

    if (passwordEchoEnabled)
        protectedDocument()->updateLayoutIgnorePendingStylesheets();

    auto node = protectedNode();
    if (!node->hasEditableStyle())
        return;

    if (passwordEchoEnabled) {
        if (CheckedPtr renderText = node->renderer())
            renderText->momentarilyRevealLastTypedCharacter(m_offset + m_text.length());
    }
    
    node->insertData(m_offset, m_text);
}

void InsertIntoTextNodeCommand::doReapply()
{
    auto node = protectedNode();
    if (!node->hasEditableStyle())
        return;

    node->insertData(m_offset, m_text);
}

void InsertIntoTextNodeCommand::doUnapply()
{
    auto node = protectedNode();
    if (!node->hasEditableStyle())
        return;

    node->deleteData(m_offset, m_text.length());
}

#ifndef NDEBUG

void InsertIntoTextNodeCommand::getNodesInCommand(NodeSet& nodes)
{
    auto node = protectedNode();
    addNodeAndDescendants(node.ptr(), nodes);
}

#endif

} // namespace WebCore
