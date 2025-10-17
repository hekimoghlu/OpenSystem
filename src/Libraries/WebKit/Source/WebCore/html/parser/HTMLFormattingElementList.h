/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 25, 2022.
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

#include "HTMLStackItem.h"
#include <wtf/Forward.h>
#include <wtf/RefPtr.h>
#include <wtf/Vector.h>

namespace WebCore {

class Element;

// This may end up merged into HTMLElementStack.
class HTMLFormattingElementList {
    WTF_MAKE_NONCOPYABLE(HTMLFormattingElementList);
public:
    HTMLFormattingElementList();
    ~HTMLFormattingElementList();

    // Ideally Entry would be private, but HTMLTreeBuilder has to coordinate
    // between the HTMLFormattingElementList and HTMLElementStack and needs
    // access to Entry::isMarker() and Entry::replaceElement() to do so.
    class Entry {
    public:
        // Inline because they're hot and Vector<T> uses them.
        explicit Entry(HTMLStackItem&& item)
            : m_item(WTFMove(item))
        {
        }
        enum MarkerEntryType { MarkerEntry };
        Entry(MarkerEntryType)
        {
        }

        bool isMarker() const { return m_item.isNull(); }

        const HTMLStackItem& stackItem() const { return m_item; }
        Element& element() const
        {
            // Callers should check isMarker() before calling element().
            return m_item.element();
        }
        void replaceElement(HTMLStackItem&& item) { m_item = WTFMove(item); }

        // Needed for use with Vector. This is super-hot and must be inline.
        bool operator==(Element* element) const { return m_item.elementOrNull() == element; }

    private:
        HTMLStackItem m_item;
    };

    class Bookmark {
    public:
        explicit Bookmark(Entry& entry)
            : m_hasBeenMoved(false)
            , m_mark(&entry)
        {
        }

        void moveToAfter(Entry& before)
        {
            m_hasBeenMoved = true;
            m_mark = &before;
        }

        bool hasBeenMoved() const { return m_hasBeenMoved; }
        Entry& mark() const { ASSERT(m_mark); return *m_mark; }

    private:
        friend class HTMLFormattingElementList;
        bool m_hasBeenMoved;
        Entry* m_mark;
    };

    bool isEmpty() const { return !size(); }
    size_t size() const { return m_entries.size(); }

    Element* closestElementInScopeWithName(ElementName);

    Entry* find(Element&);
    bool contains(Element&);
    void append(HTMLStackItem&&);
    void remove(Element&);
    void removeUpdatingBookmark(Element&, Bookmark&);

    Bookmark bookmarkFor(Element&);
    void swapTo(Ref<Element> oldElement, HTMLStackItem&& newItem, const Bookmark&);

    void appendMarker();
    // clearToLastMarker also clears the marker (per the HTML5 spec).
    void clearToLastMarker();

    const Entry& at(size_t i) const { return m_entries[i]; }
    Entry& at(size_t i) { return m_entries[i]; }

#if ENABLE(TREE_DEBUGGING)
    void show();
#endif

private:
    Entry& first() { return at(0); }

    // http://www.whatwg.org/specs/web-apps/current-work/multipage/parsing.html#list-of-active-formatting-elements
    // These functions enforce the "Noah's Ark" condition, which removes redundant mis-nested elements.
    Vector<const HTMLStackItem*> tryToEnsureNoahsArkConditionQuickly(HTMLStackItem&);
    void ensureNoahsArkCondition(HTMLStackItem&);

    Vector<Entry> m_entries;
};

} // namespace WebCore
