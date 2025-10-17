/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 10, 2022.
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

#include "CollectionTraversal.h"
#include "ElementChildIteratorInlines.h"
#include "HTMLOptionsCollectionInlines.h"
#include "TypedElementDescendantIteratorInlines.h"

namespace WebCore {

// CollectionTraversal::Descendants

template <typename CollectionClass>
inline auto CollectionTraversal<CollectionTraversalType::Descendants>::begin(const CollectionClass& collection, ContainerNode& rootNode) -> Iterator
{
    auto it = descendantsOfType<Element>(rootNode).begin();
    while (it && !collection.elementMatches(*it))
        ++it;
    // Drop iterator assertions because HTMLCollections / NodeList use a fine-grained invalidation scheme.
    it.dropAssertions();
    return it;
}

template <typename CollectionClass>
inline auto CollectionTraversal<CollectionTraversalType::Descendants>::last(const CollectionClass& collection, ContainerNode& rootNode) -> Iterator
{
    Iterator it { rootNode, ElementTraversal::lastWithin(rootNode) };
    while (it && !collection.elementMatches(*it))
        --it;
    // Drop iterator assertions because HTMLCollections / NodeList use a fine-grained invalidation scheme.
    it.dropAssertions();
    return it;
}

template <typename CollectionClass>
inline void CollectionTraversal<CollectionTraversalType::Descendants>::traverseForward(const CollectionClass& collection, Iterator& current, unsigned count, unsigned& traversedCount)
{
    ASSERT(collection.elementMatches(*current));
    for (traversedCount = 0; traversedCount < count; ++traversedCount) {
        do {
            ++current;
            if (!current)
                return;
        } while (!collection.elementMatches(*current));
    }
}

template <typename CollectionClass>
inline void CollectionTraversal<CollectionTraversalType::Descendants>::traverseBackward(const CollectionClass& collection, Iterator& current, unsigned count)
{
    ASSERT(collection.elementMatches(*current));
    for (; count; --count) {
        do {
            --current;
            if (!current)
                return;
        } while (!collection.elementMatches(*current));
    }
}

// CollectionTraversal::ChildrenOnly

template <typename CollectionClass>
inline auto CollectionTraversal<CollectionTraversalType::ChildrenOnly>::begin(const CollectionClass& collection, ContainerNode& rootNode) -> Iterator
{
    auto it = childrenOfType<Element>(rootNode).begin();
    while (it && !collection.elementMatches(*it))
        ++it;
    // Drop iterator assertions because HTMLCollections / NodeList use a fine-grained invalidation scheme.
    it.dropAssertions();
    return it;
}

template <typename CollectionClass>
inline auto CollectionTraversal<CollectionTraversalType::ChildrenOnly>::last(const CollectionClass& collection, ContainerNode& rootNode) -> Iterator
{
    auto* lastElement = childrenOfType<Element>(rootNode).last();
    if (!lastElement)
        return childrenOfType<Element>(rootNode).begin();
    auto it = childrenOfType<Element>(rootNode).beginAt(*lastElement);
    while (it && !collection.elementMatches(*it))
        --it;
    // Drop iterator assertions because HTMLCollections / NodeList use a fine-grained invalidation scheme.
    it.dropAssertions();
    return it;
}

template <typename CollectionClass>
inline void CollectionTraversal<CollectionTraversalType::ChildrenOnly>::traverseForward(const CollectionClass& collection, Iterator& current, unsigned count, unsigned& traversedCount)
{
    ASSERT(collection.elementMatches(*current));
    for (traversedCount = 0; traversedCount < count; ++traversedCount) {
        do {
            ++current;
            if (!current)
                return;
        } while (!collection.elementMatches(*current));
    }
}

template <typename CollectionClass>
inline void CollectionTraversal<CollectionTraversalType::ChildrenOnly>::traverseBackward(const CollectionClass& collection, Iterator& current, unsigned count)
{
    ASSERT(collection.elementMatches(*current));
    for (; count; --count) {
        do {
            --current;
            if (!current)
                return;
        } while (!collection.elementMatches(*current));
    }
}

// CollectionTraversal::CustomForwardOnly

template <typename CollectionClass>
inline Element* CollectionTraversal<CollectionTraversalType::CustomForwardOnly>::begin(const CollectionClass& collection, ContainerNode&)
{
    return collection.customElementAfter(nullptr);
}

template <typename CollectionClass>
inline Element* CollectionTraversal<CollectionTraversalType::CustomForwardOnly>::last(const CollectionClass&, ContainerNode&)
{
    ASSERT_NOT_REACHED();
    return nullptr;
}

template <typename CollectionClass>
inline void CollectionTraversal<CollectionTraversalType::CustomForwardOnly>::traverseForward(const CollectionClass& collection, Element*& current, unsigned count, unsigned& traversedCount)
{
    Element* element = current;
    for (traversedCount = 0; traversedCount < count; ++traversedCount) {
        element = collection.customElementAfter(element);
        if (!element) {
            current = nullptr;
            return;
        }
    }
    current = element;
}

template <typename CollectionClass>
inline void CollectionTraversal<CollectionTraversalType::CustomForwardOnly>::traverseBackward(const CollectionClass&, Element*&, unsigned count)
{
    UNUSED_PARAM(count);
    ASSERT_NOT_REACHED();
}


}
