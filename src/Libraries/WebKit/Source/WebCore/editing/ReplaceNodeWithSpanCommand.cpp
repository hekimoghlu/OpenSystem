/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 27, 2025.
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
#include "ReplaceNodeWithSpanCommand.h"

#include "Editing.h"
#include "HTMLSpanElement.h"

namespace WebCore {

ReplaceNodeWithSpanCommand::ReplaceNodeWithSpanCommand(Ref<HTMLElement>&& element)
    : SimpleEditCommand(element->document())
    , m_elementToReplace(WTFMove(element))
{
}

static void swapInNodePreservingAttributesAndChildren(Ref<HTMLElement> newNode, HTMLElement& nodeToReplace)
{
    ASSERT(nodeToReplace.isConnected());
    RefPtr parentNode = nodeToReplace.parentNode();

    // FIXME: Fix this to send the proper MutationRecords when MutationObservers are present.
    newNode->cloneDataFromElement(nodeToReplace);
    NodeVector children;
    collectChildNodes(nodeToReplace, children);
    for (auto& child : children)
        newNode->appendChild(child);

    parentNode->insertBefore(newNode, &nodeToReplace);
    parentNode->removeChild(nodeToReplace);
}

void ReplaceNodeWithSpanCommand::doApply()
{
    if (!m_elementToReplace->isConnected())
        return;
    if (!m_spanElement)
        m_spanElement = HTMLSpanElement::create(m_elementToReplace->document());
    swapInNodePreservingAttributesAndChildren(protectedSpanElement().releaseNonNull(), protectedElementToReplace());
}

void ReplaceNodeWithSpanCommand::doUnapply()
{
    RefPtr spanElement = protectedSpanElement();
    if (!spanElement || !spanElement->isConnected())
        return;
    swapInNodePreservingAttributesAndChildren(protectedElementToReplace(), *spanElement);
}

#ifndef NDEBUG
void ReplaceNodeWithSpanCommand::getNodesInCommand(NodeSet& nodes)
{
    addNodeAndDescendants(m_elementToReplace.ptr(), nodes);
    addNodeAndDescendants(m_spanElement.get(), nodes);
}
#endif

} // namespace WebCore
