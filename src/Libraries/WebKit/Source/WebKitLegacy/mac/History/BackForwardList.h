/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 8, 2022.
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

#include <WebCore/BackForwardClient.h>
#include <WebCore/BackForwardFrameItemIdentifier.h>
#include <WebCore/BackForwardItemIdentifier.h>
#include <WebCore/FrameIdentifier.h>
#include <wtf/HashSet.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

OBJC_CLASS WebView;

typedef HashSet<RefPtr<WebCore::HistoryItem>> HistoryItemHashSet;

class BackForwardList : public WebCore::BackForwardClient, public CanMakeWeakPtr<BackForwardList> {
public: 
    static Ref<BackForwardList> create(WebView *webView) { return adoptRef(*new BackForwardList(webView)); }
    virtual ~BackForwardList();

    WebView *webView() { return m_webView; }

    void addItem(Ref<WebCore::HistoryItem>&&) override;
    void setChildItem(WebCore::BackForwardFrameItemIdentifier, Ref<WebCore::HistoryItem>&&) final { }
    void goBack();
    void goForward();
    void goToItem(WebCore::HistoryItem&) override;

    RefPtr<WebCore::HistoryItem> backItem();
    RefPtr<WebCore::HistoryItem> currentItem();
    RefPtr<WebCore::HistoryItem> forwardItem();
    RefPtr<WebCore::HistoryItem> itemAtIndex(int, WebCore::FrameIdentifier) override;

    void backListWithLimit(int, Vector<Ref<WebCore::HistoryItem>>&);
    void forwardListWithLimit(int, Vector<Ref<WebCore::HistoryItem>>&);

    int capacity();
    void setCapacity(int);
    bool enabled();
    void setEnabled(bool);
    unsigned backListCount() const override;
    unsigned forwardListCount() const override;
    bool containsItem(const WebCore::HistoryItem&) const final;

    void close() override;
    bool closed();

    void removeItem(WebCore::HistoryItem&);
    const Vector<Ref<WebCore::HistoryItem>>& entries() const { return m_entries; }

#if PLATFORM(IOS_FAMILY)
    unsigned current();
    void setCurrent(unsigned newCurrent);
#endif

private:
    explicit BackForwardList(WebView *);

    WebView* m_webView;
    Vector<Ref<WebCore::HistoryItem>> m_entries;
    HistoryItemHashSet m_entryHash;
    unsigned m_current;
    unsigned m_capacity;
    bool m_closed;
    bool m_enabled;
};
