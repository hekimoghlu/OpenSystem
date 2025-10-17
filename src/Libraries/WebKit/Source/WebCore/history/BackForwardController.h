/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 17, 2024.
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

#include "BackForwardFrameItemIdentifier.h"
#include "FrameIdentifier.h"
#include <wtf/CheckedPtr.h>
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>

namespace WebCore {

class BackForwardClient;
class HistoryItem;
class Page;

class BackForwardController final : public CanMakeCheckedPtr<BackForwardController> {
    WTF_MAKE_TZONE_ALLOCATED(BackForwardController);
    WTF_MAKE_NONCOPYABLE(BackForwardController);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(BackForwardController);
public:
    BackForwardController(Page&, Ref<BackForwardClient>&&);
    ~BackForwardController();

    BackForwardClient& client() { return m_client.get(); }
    const BackForwardClient& client() const { return m_client.get(); }

    WEBCORE_EXPORT bool canGoBackOrForward(int distance) const;
    void goBackOrForward(int distance);

    WEBCORE_EXPORT bool goBack();
    WEBCORE_EXPORT bool goForward();

    void addItem(Ref<HistoryItem>&&);
    void setChildItem(BackForwardFrameItemIdentifier, Ref<HistoryItem>&&);
    void setCurrentItem(HistoryItem&);

    unsigned count() const;
    WEBCORE_EXPORT unsigned backCount() const;
    WEBCORE_EXPORT unsigned forwardCount() const;

    WEBCORE_EXPORT RefPtr<HistoryItem> itemAtIndex(int, std::optional<FrameIdentifier> = std::nullopt);
    bool containsItem(const HistoryItem&) const;

    void close();

    WEBCORE_EXPORT RefPtr<HistoryItem> backItem(std::optional<FrameIdentifier> = std::nullopt);
    WEBCORE_EXPORT RefPtr<HistoryItem> currentItem(std::optional<FrameIdentifier> = std::nullopt);
    WEBCORE_EXPORT RefPtr<HistoryItem> forwardItem(std::optional<FrameIdentifier> = std::nullopt);

    Vector<Ref<HistoryItem>> allItems();

private:
    Ref<Page> protectedPage() const;
    Ref<BackForwardClient> protectedClient() const;

    WeakRef<Page> m_page;
    Ref<BackForwardClient> m_client;
};

} // namespace WebCore
