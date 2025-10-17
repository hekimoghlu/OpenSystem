/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 21, 2024.
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

#include "CachedHTMLCollection.h"
#include "CollectionIndexCacheInlines.h"
#include "CollectionTraversalInlines.h"
#include "HTMLCollectionInlines.h"
#include "TreeScopeInlines.h"

namespace WebCore {

template <typename HTMLCollectionClass, CollectionTraversalType traversalType>
CachedHTMLCollection<HTMLCollectionClass, traversalType>::~CachedHTMLCollection()
{
    if (m_indexCache.hasValidCache())
        document().unregisterCollection(*this);
}

template <typename HTMLCollectionClass, CollectionTraversalType traversalType>
unsigned CachedHTMLCollection<HTMLCollectionClass, traversalType>::length() const
{
    return m_indexCache.nodeCount(collection());
}

template <typename HTMLCollectionClass, CollectionTraversalType traversalType>
Element* CachedHTMLCollection<HTMLCollectionClass, traversalType>::item(unsigned offset) const
{
    return m_indexCache.nodeAt(collection(), offset);
}

template <typename HTMLCollectionClass, CollectionTraversalType traversalType>
size_t CachedHTMLCollection<HTMLCollectionClass, traversalType>::memoryCost() const
{
    // memoryCost() may be invoked concurrently from a GC thread, and we need to be careful about what data we access here and how.
    // Accessing m_indexCache.memoryCost() is safe because because it doesn't involve any pointer chasing.
    // HTMLCollection::memoryCost() ensures its own thread safety.
    return m_indexCache.memoryCost() + HTMLCollection::memoryCost();
}

template <typename HTMLCollectionClass, CollectionTraversalType traversalType>
void CachedHTMLCollection<HTMLCollectionClass, traversalType>::invalidateCacheForDocument(Document& document)
{
    HTMLCollection::invalidateCacheForDocument(document);
    if (m_indexCache.hasValidCache()) {
        document.unregisterCollection(*this);
        m_indexCache.invalidate();
    }
}

static inline bool nameShouldBeVisibleInDocumentAll(HTMLElement& element)
{
    // https://html.spec.whatwg.org/multipage/infrastructure.html#all-named-elements
    return element.hasTagName(HTMLNames::aTag)
        || element.hasTagName(HTMLNames::buttonTag)
        || element.hasTagName(HTMLNames::embedTag)
        || element.hasTagName(HTMLNames::formTag)
        || element.hasTagName(HTMLNames::frameTag)
        || element.hasTagName(HTMLNames::framesetTag)
        || element.hasTagName(HTMLNames::iframeTag)
        || element.hasTagName(HTMLNames::imgTag)
        || element.hasTagName(HTMLNames::inputTag)
        || element.hasTagName(HTMLNames::mapTag)
        || element.hasTagName(HTMLNames::metaTag)
        || element.hasTagName(HTMLNames::objectTag)
        || element.hasTagName(HTMLNames::selectTag)
        || element.hasTagName(HTMLNames::textareaTag);
}

static inline bool nameShouldBeVisibleInDocumentAll(Element& element)
{
    auto* htmlElement = dynamicDowncast<HTMLElement>(element);
    return htmlElement && nameShouldBeVisibleInDocumentAll(*htmlElement);
}

template <typename HTMLCollectionClass, CollectionTraversalType traversalType>
Element* CachedHTMLCollection<HTMLCollectionClass, traversalType>::namedItem(const AtomString& name) const
{
    // http://msdn.microsoft.com/workshop/author/dhtml/reference/methods/nameditem.asp
    // This method first searches for an object with a matching id
    // attribute. If a match is not found, the method then searches for an
    // object with a matching name attribute, but only on those elements
    // that are allowed a name attribute.

    if (name.isEmpty())
        return nullptr;

    ContainerNode& root = rootNode();
    if (traversalType != CollectionTraversalType::CustomForwardOnly && root.isInTreeScope()) {
        RefPtr<Element> candidate;

        TreeScope& treeScope = root.treeScope();
        if (treeScope.hasElementWithId(name)) {
            if (!treeScope.containsMultipleElementsWithId(name))
                candidate = treeScope.getElementById(name);
        } else if (treeScope.hasElementWithName(name)) {
            if (!treeScope.containsMultipleElementsWithName(name)) {
                if ((candidate = treeScope.getElementByName(name))) {
                    if (!is<HTMLElement>(*candidate))
                        candidate = nullptr;
                    else if (type() == CollectionType::DocAll && !nameShouldBeVisibleInDocumentAll(*candidate))
                        candidate = nullptr;
                }
            }
        } else
            return nullptr;

        if (candidate && collection().elementMatches(*candidate)) {
            if (traversalType == CollectionTraversalType::ChildrenOnly ? candidate->parentNode() == &root : candidate->isDescendantOf(root))
                return candidate.get();
        }
    }

    return namedItemSlow(name);
}

template <typename HTMLCollectionClass, CollectionTraversalType traversalType>
auto CachedHTMLCollection<HTMLCollectionClass, traversalType>::collectionBegin() const -> Iterator
{
    return Traversal::begin(collection(), rootNode());
}

template <typename HTMLCollectionClass, CollectionTraversalType traversalType>
auto CachedHTMLCollection<HTMLCollectionClass, traversalType>::collectionLast() const -> Iterator
{
    return Traversal::last(collection(), rootNode());
}

template <typename HTMLCollectionClass, CollectionTraversalType traversalType>
void CachedHTMLCollection<HTMLCollectionClass, traversalType>::collectionTraverseForward(Iterator& current, unsigned count, unsigned& traversedCount) const
{
    Traversal::traverseForward(collection(), current, count, traversedCount);
}

template <typename HTMLCollectionClass, CollectionTraversalType traversalType>
void CachedHTMLCollection<HTMLCollectionClass, traversalType>::collectionTraverseBackward(Iterator& current, unsigned count) const
{
    Traversal::traverseBackward(collection(), current, count);
}

template <typename HTMLCollectionClass, CollectionTraversalType traversalType>
bool CachedHTMLCollection<HTMLCollectionClass, traversalType>::collectionCanTraverseBackward() const
{
    return traversalType != CollectionTraversalType::CustomForwardOnly;
}

}
