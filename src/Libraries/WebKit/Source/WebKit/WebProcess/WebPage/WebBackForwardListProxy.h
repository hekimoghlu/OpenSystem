/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 7, 2023.
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

#include "WebBackForwardListCounts.h"
#include <WebCore/BackForwardClient.h>
#include <WebCore/BackForwardItemIdentifier.h>
#include <WebCore/FrameIdentifier.h>
#include <WebCore/PageIdentifier.h>

namespace WebKit {

class WebPage;

class WebBackForwardListProxy : public WebCore::BackForwardClient {
public: 
    static Ref<WebBackForwardListProxy> create(WebPage& page) { return adoptRef(*new WebBackForwardListProxy(page)); }

    static void removeItem(WebCore::BackForwardItemIdentifier);

    void clearCachedListCounts();

private:
    WebBackForwardListProxy(WebPage&);

    void addItem(Ref<WebCore::HistoryItem>&&) override;
    void setChildItem(WebCore::BackForwardFrameItemIdentifier, Ref<WebCore::HistoryItem>&&) final;

    void goToItem(WebCore::HistoryItem&) override;

    RefPtr<WebCore::HistoryItem> itemAtIndex(int, WebCore::FrameIdentifier) override;
    unsigned backListCount() const override;
    unsigned forwardListCount() const override;
    bool containsItem(const WebCore::HistoryItem&) const final;
    const WebBackForwardListCounts& cacheListCountsIfNecessary() const;

    void close() override;

    WeakPtr<WebPage> m_page;
    mutable std::optional<WebBackForwardListCounts> m_cachedBackForwardListCounts;
};

} // namespace WebKit
