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
#pragma once

#include "CollectionType.h"
#include "ElementChildIterator.h"
#include "TypedElementDescendantIterator.h"

namespace WebCore {

class ContainerNode;

template <CollectionTraversalType traversalType>
struct CollectionTraversal { };

template <>
struct CollectionTraversal<CollectionTraversalType::Descendants> {
    using Iterator = ElementDescendantIterator<Element>;

    template <typename CollectionClass>
    static inline Iterator begin(const CollectionClass&, ContainerNode& rootNode);

    template <typename CollectionClass>
    static inline Iterator last(const CollectionClass&, ContainerNode& rootNode);

    template <typename CollectionClass>
    static inline void traverseForward(const CollectionClass&, Iterator& current, unsigned count, unsigned& traversedCount);

    template <typename CollectionClass>
    static inline void traverseBackward(const CollectionClass&, Iterator& current, unsigned count);
};

template <>
struct CollectionTraversal<CollectionTraversalType::ChildrenOnly> {
    using Iterator = ElementChildIterator<Element>;

    template <typename CollectionClass>
    static inline Iterator begin(const CollectionClass&, ContainerNode& rootNode);

    template <typename CollectionClass>
    static inline Iterator last(const CollectionClass&, ContainerNode& rootNode);

    template <typename CollectionClass>
    static inline void traverseForward(const CollectionClass&, Iterator& current, unsigned count, unsigned& traversedCount);

    template <typename CollectionClass>
    static inline void traverseBackward(const CollectionClass&, Iterator& current, unsigned count);
};

template <>
struct CollectionTraversal<CollectionTraversalType::CustomForwardOnly> {
    using Iterator = Element*;

    static constexpr Element* end(ContainerNode&) { return nullptr; }

    template <typename CollectionClass>
    static inline Element* begin(const CollectionClass&, ContainerNode&);

    template <typename CollectionClass>
    static inline Element* last(const CollectionClass&, ContainerNode&);

    template <typename CollectionClass>
    static inline void traverseForward(const CollectionClass&, Element*& current, unsigned count, unsigned& traversedCount);

    template <typename CollectionClass>
    static inline void traverseBackward(const CollectionClass&, Element*&, unsigned count);
};


} // namespace WebCore
