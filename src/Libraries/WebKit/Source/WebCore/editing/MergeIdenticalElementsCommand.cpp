/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 31, 2024.
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
#include "MergeIdenticalElementsCommand.h"

#include "Element.h"

namespace WebCore {

MergeIdenticalElementsCommand::MergeIdenticalElementsCommand(Ref<Element>&& first, Ref<Element>&& second)
    : SimpleEditCommand(first->document())
    , m_element1(WTFMove(first))
    , m_element2(WTFMove(second))
{
    ASSERT(m_element1->nextSibling() == m_element2.ptr());
}

void MergeIdenticalElementsCommand::doApply()
{
    Ref element1 = protectedElement1();
    Ref element2 = protectedElement2();
    if (element1->nextSibling() != element2.ptr() || !element1->hasEditableStyle() || !element2->hasEditableStyle())
        return;

    m_atChild = element2->firstChild();

    Vector<Ref<Node>> children;
    for (Node* child = element1->firstChild(); child; child = child->nextSibling())
        children.append(*child);

    for (auto& child : children)
        element2->insertBefore(child, m_atChild.copyRef());

    element1->remove();
}

void MergeIdenticalElementsCommand::doUnapply()
{
    RefPtr<Node> atChild = WTFMove(m_atChild);

    Ref element2 = protectedElement2();
    RefPtr parent = element2->parentNode();
    if (!parent || !parent->hasEditableStyle())
        return;

    Ref element1 = protectedElement1();
    if (parent->insertBefore(element1, element2.copyRef()).hasException())
        return;

    Vector<Ref<Node>> children;
    for (Node* child = element2->firstChild(); child && child != atChild; child = child->nextSibling())
        children.append(*child);

    for (auto& child : children)
        element1->appendChild(child);
}

#ifndef NDEBUG
void MergeIdenticalElementsCommand::getNodesInCommand(NodeSet& nodes)
{
    addNodeAndDescendants(m_element1.ptr(), nodes);
    addNodeAndDescendants(m_element2.ptr(), nodes);
}
#endif

} // namespace WebCore
