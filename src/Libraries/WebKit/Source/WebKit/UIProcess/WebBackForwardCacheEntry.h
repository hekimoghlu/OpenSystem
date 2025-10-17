/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 3, 2025.
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

#include <WebCore/BackForwardItemIdentifier.h>
#include <WebCore/ProcessIdentifier.h>
#include <wtf/CheckedRef.h>
#include <wtf/Forward.h>
#include <wtf/RefCounted.h>
#include <wtf/RunLoop.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {

class SuspendedPageProxy;
class WebBackForwardCache;
class WebProcessProxy;

class WebBackForwardCacheEntry : public RefCounted<WebBackForwardCacheEntry> {
    WTF_MAKE_TZONE_ALLOCATED(WebBackForwardCacheEntry);
public:
    static Ref<WebBackForwardCacheEntry> create(WebBackForwardCache&, WebCore::BackForwardItemIdentifier, WebCore::ProcessIdentifier, RefPtr<SuspendedPageProxy>&&);
    ~WebBackForwardCacheEntry();

    WebBackForwardCache* backForwardCache() const;

    SuspendedPageProxy* suspendedPage() const { return m_suspendedPage.get(); }
    Ref<SuspendedPageProxy> takeSuspendedPage();
    WebCore::ProcessIdentifier processIdentifier() const { return m_processIdentifier; }
    RefPtr<WebProcessProxy> process() const;

private:
    WebBackForwardCacheEntry(WebBackForwardCache&, WebCore::BackForwardItemIdentifier, WebCore::ProcessIdentifier, RefPtr<SuspendedPageProxy>&&);

    void expirationTimerFired();

    WeakPtr<WebBackForwardCache> m_backForwardCache;
    WebCore::ProcessIdentifier m_processIdentifier;
    Markable<WebCore::BackForwardItemIdentifier> m_backForwardItemID;
    RefPtr<SuspendedPageProxy> m_suspendedPage;
    RunLoop::Timer m_expirationTimer;
};

} // namespace WebKit
