/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 21, 2022.
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
#include "SplitElementCommand.h"

#include "CompositeEditCommand.h"
#include "Element.h"
#include "ElementInlines.h"
#include "HTMLNames.h"
#include <wtf/Assertions.h>

namespace WebCore {

SplitElementCommand::SplitElementCommand(Ref<Element>&& element, Ref<Node>&& atChild)
    : SimpleEditCommand(element->document())
    , m_element2(WTFMove(element))
    , m_atChild(WTFMove(atChild))
{
    ASSERT(m_atChild->parentNode() == m_element2.ptr());
}

void SplitElementCommand::executeApply()
{
    if (m_atChild->parentNode() != m_element2.ptr())
        return;
    
    Ref element2 = m_element2;
    Vector<Ref<Node>> children;
    for (RefPtr node = element2->firstChild(); node != m_atChild.ptr(); node = node->nextSibling())
        children.append(*node);

    RefPtr parent = element2->parentNode();
    if (!parent || !parent->hasEditableStyle())
        return;
    RefPtr element1 = m_element1;
    if (parent->insertBefore(*element1, element2.copyRef()).hasException())
        return;

    // Delete id attribute from the second element because the same id cannot be used for more than one element
    element2->removeAttribute(HTMLNames::idAttr);

    for (auto& child : children)
        element1->appendChild(child);
}
    
void SplitElementCommand::doApply()
{
    m_element1 = protectedElement2()->cloneElementWithoutChildren(protectedDocument());
    
    executeApply();
}

void SplitElementCommand::doUnapply()
{
    RefPtr element1 = m_element1;
    Ref element2 = m_element2;
    if (!element1 || !element1->hasEditableStyle() || !element2->hasEditableStyle())
        return;

    Vector<Ref<Node>> children;
    for (Node* node = element1->firstChild(); node; node = node->nextSibling())
        children.append(*node);

    RefPtr<Node> refChild = element2->firstChild();

    for (auto& child : children)
        element2->insertBefore(child, refChild.copyRef());

    // Recover the id attribute of the original element.
    const AtomString& id = element1->getIdAttribute();
    if (!id.isNull())
        element2->setIdAttribute(id);

    element1->remove();
}

void SplitElementCommand::doReapply()
{
    if (!m_element1)
        return;
    
    executeApply();
}

#ifndef NDEBUG
void SplitElementCommand::getNodesInCommand(NodeSet& nodes)
{
    addNodeAndDescendants(protectedElement1().get(), nodes);
    addNodeAndDescendants(protectedElement2().ptr(), nodes);
    addNodeAndDescendants(Ref { m_atChild }.ptr(), nodes);
}
#endif
    
} // namespace WebCore
