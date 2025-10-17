/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 28, 2023.
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
#include "TreeWalker.h"

#include "ContainerNode.h"
#include "NodeTraversal.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(TreeWalker);

TreeWalker::TreeWalker(Node& rootNode, unsigned long whatToShow, RefPtr<NodeFilter>&& filter)
    : NodeIteratorBase(rootNode, whatToShow, WTFMove(filter))
    , m_current(root())
{
}

void TreeWalker::setCurrentNode(Node& node)
{
    m_current = node;
}

inline Node* TreeWalker::setCurrent(Ref<Node>&& node)
{
    m_current = WTFMove(node);
    return m_current.ptr();
}

ExceptionOr<Node*> TreeWalker::parentNode()
{
    RefPtr node = m_current.ptr();
    while (node != &root()) {
        node = node->parentNode();
        if (!node)
            return nullptr;

        auto filterResult = acceptNode(*node);
        if (filterResult.hasException())
            return filterResult.releaseException();

        if (filterResult.returnValue() == NodeFilter::FILTER_ACCEPT)
            return setCurrent(node.releaseNonNull());
    }
    return nullptr;
}

ExceptionOr<Node*> TreeWalker::firstChild()
{
    for (RefPtr node = m_current->firstChild(); node; ) {
        auto filterResult = acceptNode(*node);
        if (filterResult.hasException())
            return filterResult.releaseException();

        switch (filterResult.returnValue()) {
            case NodeFilter::FILTER_ACCEPT:
                m_current = node.releaseNonNull();
                return m_current.ptr();
            case NodeFilter::FILTER_SKIP:
                if (node->firstChild()) {
                    node = node->firstChild();
                    continue;
                }
                break;
            case NodeFilter::FILTER_REJECT:
                break;
        }
        do {
            if (node->nextSibling()) {
                node = node->nextSibling();
                break;
            }
            RefPtr parent = node->parentNode();
            if (!parent || parent == &root() || parent == m_current.ptr())
                return nullptr;
            node = WTFMove(parent);
        } while (node);
    }
    return nullptr;
}

ExceptionOr<Node*> TreeWalker::lastChild()
{
    for (RefPtr node = m_current->lastChild(); node; ) {
        auto filterResult = acceptNode(*node);
        if (filterResult.hasException())
            return filterResult.releaseException();

        switch (filterResult.returnValue()) {
            case NodeFilter::FILTER_ACCEPT:
                m_current = node.releaseNonNull();
                return m_current.ptr();
            case NodeFilter::FILTER_SKIP:
                if (node->lastChild()) {
                    node = node->lastChild();
                    continue;
                }
                break;
            case NodeFilter::FILTER_REJECT:
                break;
        }
        do {
            if (node->previousSibling()) {
                node = node->previousSibling();
                break;
            }
            RefPtr parent = node->parentNode();
            if (!parent || parent == &root() || parent == m_current.ptr())
                return nullptr;
            node = WTFMove(parent);
        } while (node);
    }
    return nullptr;
}

template<TreeWalker::SiblingTraversalType type> ExceptionOr<Node*> TreeWalker::traverseSiblings()
{
    RefPtr node = m_current.ptr();
    if (node == &root())
        return nullptr;

    auto isNext = type == SiblingTraversalType::Next;
    while (true) {
        for (RefPtr sibling = isNext ? node->nextSibling() : node->previousSibling(); sibling; ) {
            auto filterResult = acceptNode(*sibling);
            if (filterResult.hasException())
                return filterResult.releaseException();

            if (filterResult.returnValue() == NodeFilter::FILTER_ACCEPT) {
                m_current = sibling.releaseNonNull();
                return m_current.ptr();
            }
            node = sibling;
            sibling = isNext ? sibling->firstChild() : sibling->lastChild();
            if (filterResult.returnValue() == NodeFilter::FILTER_REJECT || !sibling)
                sibling = isNext ? node->nextSibling() : node->previousSibling();
        }
        node = node->parentNode();
        if (!node || node == &root())
            return nullptr;

        auto filterResult = acceptNode(*node);
        if (filterResult.hasException())
            return filterResult.releaseException();

        if (filterResult.returnValue() == NodeFilter::FILTER_ACCEPT)
            return nullptr;
    }
}

ExceptionOr<Node*> TreeWalker::previousSibling()
{
    return traverseSiblings<SiblingTraversalType::Previous>();
}

ExceptionOr<Node*> TreeWalker::nextSibling()
{
    return traverseSiblings<SiblingTraversalType::Next>();
}

ExceptionOr<Node*> TreeWalker::previousNode()
{
    if (!filter()) {
        if (m_current.ptr() == &root())
            return nullptr;
        for (RefPtr node = NodeTraversal::previous(m_current); node; node = NodeTraversal::previous(*node)) {
            if (matchesWhatToShow(*node))
                return setCurrent(node.releaseNonNull());
            if (node == &root())
                break;
        }
        return nullptr;
    }
    RefPtr node = m_current.ptr();
    while (node != &root()) {
        while (RefPtr previousSibling = node->previousSibling()) {
            node = WTFMove(previousSibling);

            auto filterResult = acceptNode(*node);
            if (filterResult.hasException())
                return filterResult.releaseException();

            auto acceptNodeResult = filterResult.returnValue();
            if (acceptNodeResult == NodeFilter::FILTER_REJECT)
                continue;
            while (RefPtr lastChild = node->lastChild()) {
                node = WTFMove(lastChild);

                auto filterResult = acceptNode(*node);
                if (filterResult.hasException())
                    return filterResult.releaseException();

                acceptNodeResult = filterResult.returnValue();
                if (acceptNodeResult == NodeFilter::FILTER_REJECT)
                    break;
            }
            if (acceptNodeResult == NodeFilter::FILTER_ACCEPT) {
                m_current = node.releaseNonNull();
                return m_current.ptr();
            }
        }
        if (node == &root())
            return nullptr;
        RefPtr parent = node->parentNode();
        if (!parent)
            return nullptr;
        node = WTFMove(parent);

        auto filterResult = acceptNode(*node);
        if (filterResult.hasException())
            return filterResult.releaseException();

        if (filterResult.returnValue() == NodeFilter::FILTER_ACCEPT)
            return setCurrent(node.releaseNonNull());
    }
    return nullptr;
}

ExceptionOr<Node*> TreeWalker::nextNode()
{
    if (!filter()) {
        for (RefPtr node = NodeTraversal::next(m_current, &root()); node; node = NodeTraversal::next(*node, &root())) {
            if (matchesWhatToShow(*node))
                return setCurrent(node.releaseNonNull());
        }
        return nullptr;
    }
    RefPtr node = m_current.ptr();
Children:
    while (RefPtr firstChild = node->firstChild()) {
        node = WTFMove(firstChild);

        auto filterResult = acceptNode(*node);
        if (filterResult.hasException())
            return filterResult.releaseException();

        if (filterResult.returnValue() == NodeFilter::FILTER_ACCEPT)
            return setCurrent(node.releaseNonNull());
        if (filterResult.returnValue() == NodeFilter::FILTER_REJECT)
            break;
    }
    while (RefPtr nextSibling = NodeTraversal::nextSkippingChildren(*node, &root())) {
        node = WTFMove(nextSibling);

        auto filterResult = acceptNode(*node);
        if (filterResult.hasException())
            return filterResult.releaseException();

        if (filterResult.returnValue() == NodeFilter::FILTER_ACCEPT)
            return setCurrent(node.releaseNonNull());
        if (filterResult.returnValue() == NodeFilter::FILTER_SKIP)
            goto Children;
    }
    return nullptr;
}

} // namespace WebCore
