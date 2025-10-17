/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 14, 2025.
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

#include <WebCore/ProcessIdentifier.h>
#include <pal/SessionID.h>
#include <wtf/Forward.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakListHashSet.h>
#include <wtf/WeakPtr.h>

namespace WebKit {

class SuspendedPageProxy;
class WebBackForwardCacheEntry;
class WebBackForwardListItem;
class WebPageProxy;
class WebProcessPool;
class WebProcessProxy;

class WebBackForwardCache final : public CanMakeWeakPtr<WebBackForwardCache> {
    WTF_MAKE_TZONE_ALLOCATED(WebBackForwardCache);
public:
    explicit WebBackForwardCache(WebProcessPool&);
    ~WebBackForwardCache();

    void ref() const;
    void deref() const;

    void setCapacity(WebProcessPool&, unsigned);
    unsigned capacity() const { return m_capacity; }
    unsigned size() const { return m_itemsWithCachedPage.computeSize(); }

    void clear();
    void pruneToSize(unsigned);
    void removeEntriesForProcess(WebProcessProxy&);
    void removeEntriesForPage(WebPageProxy&);
    void removeEntriesForPageAndProcess(WebPageProxy&, WebProcessProxy&);
    void removeEntriesForSession(PAL::SessionID);

    void addEntry(WebBackForwardListItem&, Ref<SuspendedPageProxy>&&);
    void addEntry(WebBackForwardListItem&, WebCore::ProcessIdentifier);
    void removeEntry(WebBackForwardListItem&);
    void removeEntry(SuspendedPageProxy&);
    Ref<SuspendedPageProxy> takeSuspendedPage(WebBackForwardListItem&);

private:
    Ref<WebProcessPool> protectedProcessPool() const;

    void removeOldestEntry();
    void removeEntriesMatching(const Function<bool(WebBackForwardListItem&)>&);
    void addEntry(WebBackForwardListItem&, Ref<WebBackForwardCacheEntry>&&);

    WeakRef<WebProcessPool> m_processPool;
    unsigned m_capacity { 0 };
    WeakListHashSet<WebBackForwardListItem> m_itemsWithCachedPage;
};

} // namespace WebKit
