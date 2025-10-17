/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 15, 2023.
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

#include "APIObject.h"
#include "WebBackForwardListItem.h"
#include <WebCore/BackForwardItemIdentifier.h>
#include <wtf/Ref.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

namespace API {
class Array;
}

namespace WebKit {

class WebPageProxy;

struct BackForwardListState;
struct WebBackForwardListCounts;

class WebBackForwardList : public API::ObjectImpl<API::Object::Type::BackForwardList> {
public:
    static Ref<WebBackForwardList> create(WebPageProxy& page)
    {
        return adoptRef(*new WebBackForwardList(page));
    }
    void pageClosed();

    virtual ~WebBackForwardList();

    WebBackForwardListItem* itemForID(WebCore::BackForwardItemIdentifier);

    void addItem(Ref<WebBackForwardListItem>&&);
    void goToItem(WebBackForwardListItem&);
    void removeAllItems();
    void clear();

    WebBackForwardListItem* currentItem() const;
    RefPtr<WebBackForwardListItem> protectedCurrentItem() const;
    WebBackForwardListItem* backItem() const;
    WebBackForwardListItem* forwardItem() const;
    WebBackForwardListItem* itemAtIndex(int) const;

    WebBackForwardListItem* goBackItemSkippingItemsWithoutUserGesture() const;
    WebBackForwardListItem* goForwardItemSkippingItemsWithoutUserGesture() const;

    const BackForwardListItemVector& entries() const { return m_entries; }

    unsigned backListCount() const;
    unsigned forwardListCount() const;
    WebBackForwardListCounts counts() const;

    Ref<API::Array> backList() const;
    Ref<API::Array> forwardList() const;

    Ref<API::Array> backListAsAPIArrayWithLimit(unsigned limit) const;
    Ref<API::Array> forwardListAsAPIArrayWithLimit(unsigned limit) const;

    BackForwardListState backForwardListState(WTF::Function<bool (WebBackForwardListItem&)>&&) const;
    void restoreFromState(BackForwardListState);

    void setItemsAsRestoredFromSession();
    void setItemsAsRestoredFromSessionIf(Function<bool(WebBackForwardListItem&)>&&);

    Ref<FrameState> completeFrameStateForNavigation(Ref<FrameState>&&);

#if !LOG_DISABLED
    String loggingString();
#endif

private:
    explicit WebBackForwardList(WebPageProxy&);

    void didRemoveItem(WebBackForwardListItem&);

    RefPtr<WebPageProxy> protectedPage();

    WeakPtr<WebPageProxy> m_page;
    BackForwardListItemVector m_entries;
    std::optional<size_t> m_currentIndex;
};

} // namespace WebKit
