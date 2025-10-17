/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 3, 2024.
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
#include "RenderTreePosition.h"

#include "ComposedTreeIterator.h"
#include "PseudoElement.h"
#include "RenderInline.h"
#include "RenderObject.h"
#include "ShadowRoot.h"

namespace WebCore {

void RenderTreePosition::computeNextSibling(const Node& node)
{
    ASSERT(!node.renderer());
    if (m_hasValidNextSibling) {
#if ASSERT_ENABLED
        const unsigned oNSquaredAvoidanceLimit = 20;
        bool skipAssert = m_parent->isRenderView() || ++m_assertionLimitCounter > oNSquaredAvoidanceLimit;
        ASSERT(skipAssert || nextSiblingRenderer(node) == m_nextSibling);
#endif
        return;
    }
    m_nextSibling = nextSiblingRenderer(node);
    m_hasValidNextSibling = true;
}

void RenderTreePosition::invalidateNextSibling(const RenderObject& siblingRenderer)
{
    if (!m_hasValidNextSibling)
        return;
    if (m_nextSibling == &siblingRenderer)
        m_hasValidNextSibling = false;
}

RenderObject* RenderTreePosition::nextSiblingRenderer(const Node& node) const
{
    ASSERT(!node.renderer());

    auto* parentElement = m_parent->element();
    if (!parentElement)
        return nullptr;
    // FIXME: PlugingReplacement shadow trees are very wrong.
    if (parentElement == &node)
        return nullptr;

    Vector<Element*, 30> elementStack;

    // In the common case ancestor == parentElement immediately and this just pushes parentElement into stack.
    auto* ancestor = node.parentElementInComposedTree();
    while (true) {
        elementStack.append(ancestor);
        if (ancestor == parentElement)
            break;
        ancestor = ancestor->parentElementInComposedTree();
        ASSERT(ancestor);
    }
    elementStack.reverse();

    auto composedDescendants = composedTreeDescendants(*parentElement);

    auto initializeIteratorConsideringPseudoElements = [&] {
        if (auto* pseudoElement = dynamicDowncast<PseudoElement>(node)) {
            auto* host = pseudoElement->hostElement();
            if (node.isBeforePseudoElement()) {
                if (host != parentElement)
                    return composedDescendants.at(*host).traverseNext();
                return composedDescendants.begin();
            }
            ASSERT(node.isAfterPseudoElement());
            elementStack.removeLast();
            if (host != parentElement)
                return composedDescendants.at(*host).traverseNextSkippingChildren();
            return composedDescendants.end();
        }
        return composedDescendants.at(node).traverseNextSkippingChildren();
    };

    auto pushCheckingForAfterPseudoElementRenderer = [&] (Element& element) -> RenderElement* {
        ASSERT(!element.isPseudoElement());
        if (auto* before = element.beforePseudoElement()) {
            if (auto* renderer = before->renderer())
                return renderer;
        }
        elementStack.append(&element);
        return nullptr;
    };

    auto popCheckingForAfterPseudoElementRenderers = [&] (unsigned iteratorDepthToMatch) -> RenderElement* {
        while (elementStack.size() > iteratorDepthToMatch) {
            auto& element = *elementStack.takeLast();
            if (auto* after = element.afterPseudoElement()) {
                if (auto* renderer = after->renderer())
                    return renderer;
            }
        }
        return nullptr;
    };

    auto it = initializeIteratorConsideringPseudoElements();
    auto end = composedDescendants.end();

    while (it != end) {
        if (auto* renderer = popCheckingForAfterPseudoElementRenderers(it.depth()))
            return renderer;

        if (auto* renderer = it->renderer())
            return renderer;

        if (auto* element = dynamicDowncast<Element>(*it)) {
            if (element->hasDisplayContents()) {
                if (auto* renderer = pushCheckingForAfterPseudoElementRenderer(*element))
                    return renderer;
                it.traverseNext();
                continue;
            }
        }

        it.traverseNextSkippingChildren();
    }

    return popCheckingForAfterPseudoElementRenderers(0);
}

}
