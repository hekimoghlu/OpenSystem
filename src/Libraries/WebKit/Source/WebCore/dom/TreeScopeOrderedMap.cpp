/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 25, 2021.
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
#include "TreeScopeOrderedMap.h"

#include "ContainerNodeAlgorithms.h"
#include "ElementInlines.h"
#include "HTMLImageElement.h"
#include "HTMLLabelElement.h"
#include "HTMLMapElement.h"
#include "HTMLNameCollection.h"
#include "TreeScopeInlines.h"
#include "TypedElementDescendantIteratorInlines.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(TreeScopeOrderedMap);

using namespace HTMLNames;

void TreeScopeOrderedMap::clear()
{
    m_map.clear();
}

void TreeScopeOrderedMap::add(const AtomString& key, Element& element, const TreeScope& treeScope)
{
    ASSERT_WITH_SECURITY_IMPLICATION(!key.isNull());
    RELEASE_ASSERT_WITH_SECURITY_IMPLICATION(&element.treeScope() == &treeScope);
    ASSERT_WITH_SECURITY_IMPLICATION(treeScope.rootNode().containsIncludingShadowDOM(&element));

    if (!element.isInTreeScope())
        return;
    Map::AddResult addResult = m_map.ensure(key, [&element] {
        return MapEntry(&element);
    });
    MapEntry& entry = addResult.iterator->value;

#if ASSERT_ENABLED || ENABLE(SECURITY_ASSERTIONS)
    ASSERT_WITH_SECURITY_IMPLICATION(!entry.registeredElements.contains(&element));
    entry.registeredElements.add(element);
#endif

    if (addResult.isNewEntry)
        return;

    RELEASE_ASSERT_WITH_SECURITY_IMPLICATION(entry.count);
    entry.element = nullptr;
    entry.count++;
    entry.orderedList.clear();
}

void TreeScopeOrderedMap::remove(const AtomString& key, Element& element)
{
    ASSERT_WITH_SECURITY_IMPLICATION(!key.isNull());
    m_map.checkConsistency();
    auto it = m_map.find(key);

    RELEASE_ASSERT_WITH_SECURITY_IMPLICATION(it != m_map.end());

    MapEntry& entry = it->value;
    ASSERT_WITH_SECURITY_IMPLICATION(entry.registeredElements.remove(element));
    RELEASE_ASSERT_WITH_SECURITY_IMPLICATION(entry.count);
    if (entry.count == 1) {
        RELEASE_ASSERT_WITH_SECURITY_IMPLICATION(!entry.element || entry.element == &element);
        m_map.remove(it);
    } else {
        if (entry.element == &element)
            entry.element = nullptr;
        entry.count--;
        entry.orderedList.clear(); // FIXME: Remove the element instead if there are only few items left.
    }
}

template <typename KeyMatchingFunction>
inline RefPtr<Element> TreeScopeOrderedMap::get(const AtomString& key, const TreeScope& scope, const KeyMatchingFunction& keyMatches) const
{
    ASSERT_WITH_SECURITY_IMPLICATION(!key.isNull());
    m_map.checkConsistency();

    auto it = m_map.find(key);
    if (it == m_map.end())
        return nullptr;

    MapEntry& entry = it->value;
    ASSERT(entry.count);
    if (entry.element) {
        Ref element = *entry.element;
        RELEASE_ASSERT_WITH_SECURITY_IMPLICATION(&element->treeScope() == &scope);
        ASSERT_WITH_SECURITY_IMPLICATION(entry.registeredElements.contains(element.ptr()));
        return element;
    }

    // We know there's at least one node that matches; iterate to find the first one.
    Ref rootNode = scope.rootNode();
    for (Ref<Element> element : descendantsOfType<Element>(rootNode.get())) {
        if (!element->isInTreeScope())
            continue;
        if (!keyMatches(key, element))
            continue;
        entry.element = element.ptr();
        RELEASE_ASSERT_WITH_SECURITY_IMPLICATION(&element->treeScope() == &scope);
        ASSERT_WITH_SECURITY_IMPLICATION(entry.registeredElements.contains(entry.element.get()));
        return element;
    }

#if ASSERT_ENABLED
    // FormListedElement may call getElementById to find its owner form in the middle of a tree removal.
    if (auto* currentScope = ContainerChildRemovalScope::currentScope()) {
        ASSERT(&scope.rootNode() == &currentScope->parentOfRemovedTree().rootNode());
        Ref removedTree = currentScope->removedChild();
        for (Ref element : descendantsOfType<Element>(downcast<ContainerNode>(removedTree.get()))) {
            if (!keyMatches(key, element))
                continue;
            return nullptr;
        }
    }
    ASSERT_NOT_REACHED();
#endif // ASSERT_ENABLED

    return nullptr;
}

template <typename KeyMatchingFunction>
inline Vector<WeakRef<Element, WeakPtrImplWithEventTargetData>>* TreeScopeOrderedMap::getAll(const AtomString& key, const TreeScope& scope, const KeyMatchingFunction& keyMatches) const
{
    ASSERT_WITH_SECURITY_IMPLICATION(!key.isNull());
    m_map.checkConsistency();

    auto mapIterator = m_map.find(key);
    if (mapIterator == m_map.end())
        return nullptr;

    auto& entry = mapIterator->value;
    RELEASE_ASSERT_WITH_SECURITY_IMPLICATION(entry.count);

    if (entry.orderedList.isEmpty()) {
        entry.orderedList.reserveCapacity(entry.count);
        auto elementDescendants = descendantsOfType<Element>(scope.protectedRootNode().get());
        for (auto it = entry.element ? elementDescendants.beginAt(*entry.element) : elementDescendants.begin(); it; ++it) {
            if (keyMatches(key, *it))
                entry.orderedList.append(*it);
        }
        RELEASE_ASSERT_WITH_SECURITY_IMPLICATION(entry.orderedList.size() == entry.count);
    }

    return &entry.orderedList;
}

RefPtr<Element> TreeScopeOrderedMap::getElementById(const AtomString& key, const TreeScope& scope) const
{
    return get(key, scope, [] (const AtomString& key, const Element& element) {
        return element.getIdAttribute() == key;
    });
}

RefPtr<Element> TreeScopeOrderedMap::getElementByName(const AtomString& key, const TreeScope& scope) const
{
    return get(key, scope, [] (const AtomString& key, const Element& element) {
        return element.getNameAttribute() == key;
    });
}

RefPtr<HTMLMapElement> TreeScopeOrderedMap::getElementByMapName(const AtomString& key, const TreeScope& scope) const
{
    return downcast<HTMLMapElement>(get(key, scope, [] (const AtomString& key, const Element& element) {
        auto* mapElement = dynamicDowncast<HTMLMapElement>(element);
        return mapElement && mapElement->getName() == key;
    }));
}

RefPtr<HTMLImageElement> TreeScopeOrderedMap::getElementByUsemap(const AtomString& key, const TreeScope& scope) const
{
    return downcast<HTMLImageElement>(get(key, scope, [] (const AtomString& key, const Element& element) {
        // FIXME: HTML5 specification says we should match both image and object elements.
        auto* imageElement = dynamicDowncast<HTMLImageElement>(element);
        return imageElement && imageElement->matchesUsemap(key);
    }));
}

const Vector<WeakRef<Element, WeakPtrImplWithEventTargetData>>* TreeScopeOrderedMap::getElementsByLabelForAttribute(const AtomString& key, const TreeScope& scope) const
{
    return getAll(key, scope, [] (const AtomString& key, const Element& element) {
        return is<HTMLLabelElement>(element) && element.attributeWithoutSynchronization(forAttr) == key;
    });
}

RefPtr<Element> TreeScopeOrderedMap::getElementByWindowNamedItem(const AtomString& key, const TreeScope& scope) const
{
    return get(key, scope, [] (const AtomString& key, const Element& element) {
        return WindowNameCollection::elementMatches(element, key);
    });
}

RefPtr<Element> TreeScopeOrderedMap::getElementByDocumentNamedItem(const AtomString& key, const TreeScope& scope) const
{
    return get(key, scope, [] (const AtomString& key, const Element& element) {
        return DocumentNameCollection::elementMatches(element, key);
    });
}

const Vector<WeakRef<Element, WeakPtrImplWithEventTargetData>>* TreeScopeOrderedMap::getAllElementsById(const AtomString& key, const TreeScope& scope) const
{
    return getAll(key, scope, [] (const AtomString& key, const Element& element) {
        return element.getIdAttribute() == key;
    });
}

const Vector<AtomString> TreeScopeOrderedMap::keys() const
{
    return WTF::map(m_map, [](auto& entry) -> AtomString {
        return entry.key;
    });
}

} // namespace WebCore
