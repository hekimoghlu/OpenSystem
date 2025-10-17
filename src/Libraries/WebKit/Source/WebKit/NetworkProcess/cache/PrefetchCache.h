/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 4, 2022.
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

#include "PrivateRelayed.h"
#include <WebCore/ResourceRequest.h>
#include <WebCore/ResourceResponse.h>
#include <WebCore/SharedBuffer.h>
#include <WebCore/Timer.h>
#include <wtf/CheckedPtr.h>
#include <wtf/Deque.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/URLHash.h>
#include <wtf/text/WTFString.h>

namespace WebKit {

class PrefetchCache final : public CanMakeCheckedPtr<PrefetchCache> {
    WTF_MAKE_TZONE_ALLOCATED(PrefetchCache);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(PrefetchCache);
    WTF_MAKE_NONCOPYABLE(PrefetchCache);
public:
    PrefetchCache();
    ~PrefetchCache();

    void clear();

    struct Entry {
        WTF_MAKE_STRUCT_FAST_ALLOCATED;
        Entry(WebCore::ResourceResponse&&, PrivateRelayed, RefPtr<WebCore::FragmentedSharedBuffer>&&);
        Entry(WebCore::ResourceResponse&&, WebCore::ResourceRequest&&);

        Ref<WebCore::FragmentedSharedBuffer> releaseBuffer() { return buffer.releaseNonNull(); }

        WebCore::ResourceResponse response;
        PrivateRelayed privateRelayed { PrivateRelayed::No };
        // FIXME: This should probably be a variant<RefPtr, ResourceRequest> because we have one or the other but never both.
        RefPtr<WebCore::FragmentedSharedBuffer> buffer;
        WebCore::ResourceRequest redirectRequest;
    };

    std::unique_ptr<Entry> take(const URL&);
    void store(const URL&, WebCore::ResourceResponse&&, PrivateRelayed, RefPtr<WebCore::FragmentedSharedBuffer>&&);
    void storeRedirect(const URL&, WebCore::ResourceResponse&&, WebCore::ResourceRequest&&);

private:
    void clearExpiredEntries();

    using PrefetchEntriesMap = HashMap<URL, std::unique_ptr<Entry>>;
    std::unique_ptr<PrefetchEntriesMap> m_sessionPrefetches;

    using SessionPrefetchExpirationList = Deque<std::tuple<URL, WallTime>>;
    SessionPrefetchExpirationList m_sessionExpirationList;

    WebCore::Timer m_expirationTimer;
};

} // namespace WebKit
