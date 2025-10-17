/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 29, 2023.
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

class AppendNodeCommand : public SimpleEditCommand {
public:
    static Ref<AppendNodeCommand> create(Ref<ContainerNode>&& parent, Ref<Node>&& node, EditAction editingAction)
    {
        return adoptRef(*new AppendNodeCommand(WTFMove(parent), WTFMove(node), editingAction));
    }

private:
    AppendNodeCommand(Ref<ContainerNode>&& parent, Ref<Node>&&, EditAction);

    void doApply() override;
    void doUnapply() override;

#ifndef NDEBUG
    void getNodesInCommand(NodeSet&) override;
#endif

    Ref<ContainerNode> protectedParent() const { return m_parent; }
    Ref<Node> protectedNode() const { return m_node; }

    Ref<ContainerNode> m_parent;
    Ref<Node> m_node;
};

} // namespace WebCore
