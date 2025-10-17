/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 29, 2023.
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

#include "CompositeEditCommand.h"

namespace WebCore {

class SimplifyMarkupCommand : public CompositeEditCommand {
public:
    static Ref<SimplifyMarkupCommand> create(Ref<Document>&& document, Node* firstNode, Node* nodeAfterLast)
    {
        return adoptRef(*new SimplifyMarkupCommand(WTFMove(document), firstNode, nodeAfterLast));
    }

private:
    SimplifyMarkupCommand(Ref<Document>&&, Node* firstNode, Node* nodeAfterLast);

    void doApply() override;
    int pruneSubsequentAncestorsToRemove(Vector<Ref<Node>>& nodesToRemove, size_t startNodeIndex);

    RefPtr<Node> m_firstNode;
    RefPtr<Node> m_nodeAfterLast;
};

} // namespace WebCore
