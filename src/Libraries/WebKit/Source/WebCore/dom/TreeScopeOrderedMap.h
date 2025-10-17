/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 3, 2024.
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

#include "Element.h"
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

class HTMLImageElement;
class HTMLLabelElement;
class HTMLMapElement;
class TreeScope;

class TreeScopeOrderedMap {
    WTF_MAKE_TZONE_ALLOCATED(TreeScopeOrderedMap);
public:
    void add(const AtomString&, Element&, const TreeScope&);
    void remove(const AtomString&, Element&);
    void clear();

    bool contains(const AtomString&) const;
    bool containsSingle(const AtomString&) const;
    bool containsMultiple(const AtomString&) const;

    // concrete instantiations of the get<>() method template
    RefPtr<Element> getElementById(const AtomString&, const TreeScope&) const;
    RefPtr<Element> getElementByName(const AtomString&, const TreeScope&) const;
    RefPtr<HTMLMapElement> getElementByMapName(const AtomString&, const TreeScope&) const;
    RefPtr<HTMLImageElement> getElementByUsemap(const AtomString&, const TreeScope&) const;
    const Vector<WeakRef<Element, WeakPtrImplWithEventTargetData>>* getElementsByLabelForAttribute(const AtomString&, const TreeScope&) const;
    RefPtr<Element> getElementByWindowNamedItem(const AtomString&, const TreeScope&) const;
    RefPtr<Element> getElementByDocumentNamedItem(const AtomString&, const TreeScope&) const;

    const Vector<WeakRef<Element, WeakPtrImplWithEventTargetData>>* getAllElementsById(const AtomString&, const TreeScope&) const;

    const Vector<AtomString> keys() const;

private:
    template <typename KeyMatchingFunction>
    RefPtr<Element> get(const AtomString&, const TreeScope&, const KeyMatchingFunction&) const;
    template <typename KeyMatchingFunction>
    Vector<WeakRef<Element, WeakPtrImplWithEventTargetData>>* getAll(const AtomString&, const TreeScope&, const KeyMatchingFunction&) const;

    struct MapEntry {
        MapEntry() = default;
        explicit MapEntry(Element* firstElement)
            : element(firstElement)
            , count(1)
        { }

        WeakPtr<Element, WeakPtrImplWithEventTargetData> element;
        unsigned count { 0 };
        Vector<WeakRef<Element, WeakPtrImplWithEventTargetData>> orderedList;
#if ASSERT_ENABLED || ENABLE(SECURITY_ASSERTIONS)
        UncheckedKeyHashSet<WeakRef<Element, WeakPtrImplWithEventTargetData>> registeredElements;
#endif
    };

    using Map = UncheckedKeyHashMap<AtomString, MapEntry>;

    mutable Map m_map;
};

inline bool TreeScopeOrderedMap::containsSingle(const AtomString& id) const
{
    auto it = m_map.find(id);
    return it != m_map.end() && it->value.count == 1;
}

inline bool TreeScopeOrderedMap::contains(const AtomString& id) const
{
    return m_map.contains(id);
}

inline bool TreeScopeOrderedMap::containsMultiple(const AtomString& id) const
{
    auto it = m_map.find(id);
    return it != m_map.end() && it->value.count > 1;
}

} // namespace WebCore
