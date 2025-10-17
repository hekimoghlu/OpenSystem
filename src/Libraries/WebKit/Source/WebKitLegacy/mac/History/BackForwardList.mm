/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 21, 2023.
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
#import "BackForwardList.h"

#import <WebCore/BackForwardCache.h>

using namespace WebCore;

static const unsigned DefaultCapacity = 100;
static const unsigned NoCurrentItemIndex = UINT_MAX;

BackForwardList::BackForwardList(WebView *webView)
    : m_webView(webView)
    , m_current(NoCurrentItemIndex)
    , m_capacity(DefaultCapacity)
    , m_closed(true)
    , m_enabled(true)
{
}

BackForwardList::~BackForwardList()
{
    ASSERT(m_closed);
}

void BackForwardList::addItem(Ref<HistoryItem>&& newItem)
{
    if (!m_capacity || !m_enabled)
        return;
    
    // Toss anything in the forward list    
    if (m_current != NoCurrentItemIndex) {
        unsigned targetSize = m_current + 1;
        while (m_entries.size() > targetSize) {
            Ref<HistoryItem> item = m_entries.takeLast();
            m_entryHash.remove(item.ptr());
            BackForwardCache::singleton().remove(item);
        }
    }

    // Toss the first item if the list is getting too big, as long as we're not using it
    // (or even if we are, if we only want 1 entry).
    if (m_entries.size() == m_capacity && (m_current || m_capacity == 1)) {
        Ref<HistoryItem> item = WTFMove(m_entries[0]);
        m_entries.remove(0);
        m_entryHash.remove(item.ptr());
        BackForwardCache::singleton().remove(item);
        --m_current;
    }

    m_entryHash.add(newItem.ptr());
    m_entries.insert(m_current + 1, WTFMove(newItem));
    ++m_current;
}

void BackForwardList::goBack()
{
    ASSERT(m_current > 0);
    if (m_current > 0)
        m_current--;
}

void BackForwardList::goForward()
{
    ASSERT(m_current < m_entries.size() - 1);
    if (m_current < m_entries.size() - 1)
        m_current++;
}

void BackForwardList::goToItem(HistoryItem& item)
{
    if (!m_entries.size())
        return;

    unsigned index = 0;
    for (; index < m_entries.size(); ++index) {
        if (m_entries[index].ptr() == &item)
            break;
    }

    if (index < m_entries.size())
        m_current = index;
}

RefPtr<HistoryItem> BackForwardList::backItem()
{
    if (m_current && m_current != NoCurrentItemIndex)
        return m_entries[m_current - 1].copyRef();
    return nullptr;
}

RefPtr<HistoryItem> BackForwardList::currentItem()
{
    if (m_current != NoCurrentItemIndex)
        return m_entries[m_current].copyRef();
    return nullptr;
}

RefPtr<HistoryItem> BackForwardList::forwardItem()
{
    if (m_entries.size() && m_current < m_entries.size() - 1)
        return m_entries[m_current + 1].copyRef();
    return nullptr;
}

void BackForwardList::backListWithLimit(int limit, Vector<Ref<HistoryItem>>& list)
{
    list.clear();
    if (m_current != NoCurrentItemIndex) {
        unsigned first = std::max(static_cast<int>(m_current) - limit, 0);
        for (; first < m_current; ++first)
            list.append(m_entries[first].get());
    }
}

void BackForwardList::forwardListWithLimit(int limit, Vector<Ref<HistoryItem>>& list)
{
    ASSERT(limit > -1);
    list.clear();
    if (!m_entries.size())
        return;

    unsigned lastEntry = m_entries.size() - 1;
    if (m_current < lastEntry) {
        int last = std::min(m_current + limit, lastEntry);
        limit = m_current + 1;
        for (; limit <= last; ++limit)
            list.append(m_entries[limit].get());
    }
}

int BackForwardList::capacity()
{
    return m_capacity;
}

void BackForwardList::setCapacity(int size)
{    
    while (size < static_cast<int>(m_entries.size())) {
        Ref<HistoryItem> item = m_entries.takeLast();
        m_entryHash.remove(item.ptr());
        BackForwardCache::singleton().remove(item);
    }

    if (!size)
        m_current = NoCurrentItemIndex;
    else if (m_current > m_entries.size() - 1)
        m_current = m_entries.size() - 1;

    m_capacity = size;
}

bool BackForwardList::enabled()
{
    return m_enabled;
}

void BackForwardList::setEnabled(bool enabled)
{
    m_enabled = enabled;
    if (!enabled) {
        int capacity = m_capacity;
        setCapacity(0);
        setCapacity(capacity);
    }
}

unsigned BackForwardList::backListCount() const
{
    return m_current == NoCurrentItemIndex ? 0 : m_current;
}

unsigned BackForwardList::forwardListCount() const
{
    return m_current == NoCurrentItemIndex ? 0 : m_entries.size() - m_current - 1;
}

RefPtr<HistoryItem> BackForwardList::itemAtIndex(int index, WebCore::FrameIdentifier)
{
    // Do range checks without doing math on index to avoid overflow.
    if (index < -static_cast<int>(m_current))
        return nullptr;
    
    if (index > static_cast<int>(forwardListCount()))
        return nullptr;

    return m_entries[index + m_current].copyRef();
}

#if PLATFORM(IOS_FAMILY)
unsigned BackForwardList::current()
{
    return m_current;
}

void BackForwardList::setCurrent(unsigned newCurrent)
{
    m_current = newCurrent;
}
#endif

void BackForwardList::close()
{
    m_entries.clear();
    m_entryHash.clear();
    m_webView = nullptr;
    m_closed = true;
}

bool BackForwardList::closed()
{
    return m_closed;
}

void BackForwardList::removeItem(HistoryItem& item)
{
    for (unsigned i = 0; i < m_entries.size(); ++i) {
        if (m_entries[i].ptr() == &item) {
            m_entries.remove(i);
            m_entryHash.remove(&item);
            if (m_current == NoCurrentItemIndex || m_current < i)
                break;
            if (m_current > i)
                m_current--;
            else {
                size_t count = m_entries.size();
                if (m_current >= count)
                    m_current = count ? count - 1 : NoCurrentItemIndex;
            }
            break;
        }
    }
}

bool BackForwardList::containsItem(const HistoryItem& entry) const
{
    return m_entryHash.contains(const_cast<HistoryItem*>(&entry));
}
