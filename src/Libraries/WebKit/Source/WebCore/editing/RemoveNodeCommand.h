/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 15, 2024.
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

#include "EditCommand.h"

namespace WebCore {

class RemoveNodeCommand : public SimpleEditCommand {
public:
    static Ref<RemoveNodeCommand> create(Ref<Node>&& node, ShouldAssumeContentIsAlwaysEditable shouldAssumeContentIsAlwaysEditable, EditAction editingAction = EditAction::Unspecified)
    {
        return adoptRef(*new RemoveNodeCommand(WTFMove(node), shouldAssumeContentIsAlwaysEditable, editingAction));
    }

private:
    RemoveNodeCommand(Ref<Node>&&, ShouldAssumeContentIsAlwaysEditable, EditAction);

    void doApply() override;
    void doUnapply() override;

#ifndef NDEBUG
    void getNodesInCommand(NodeSet&) override;
#endif

    Ref<Node> protectedNode() const { return m_node; }

    Ref<Node> m_node;
    RefPtr<ContainerNode> m_parent;
    RefPtr<Node> m_refChild;
    ShouldAssumeContentIsAlwaysEditable m_shouldAssumeContentIsAlwaysEditable;
};

} // namespace WebCore
