/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 20, 2022.
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

#if ENABLE(NETWORK_CACHE_STALE_WHILE_REVALIDATE)

#include "NetworkCache.h"
#include "NetworkCacheEntry.h"
#include "NetworkCacheSpeculativeLoad.h"
#include <wtf/CompletionHandler.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
namespace NetworkCache {
class AsyncRevalidation;
}
}

namespace WebCore {
enum class AdvancedPrivacyProtections : uint16_t;
class ResourceRequest;
};

namespace WebKit {

class SpeculativeLoad;

namespace NetworkCache {

class AsyncRevalidation : public RefCountedAndCanMakeWeakPtr<AsyncRevalidation> {
    WTF_MAKE_TZONE_ALLOCATED(AsyncRevalidation);
public:
    enum class Result {
        Failure,
        Timeout,
        Success,
    };
    static Ref<AsyncRevalidation> create(Cache&, const GlobalFrameID&, const WebCore::ResourceRequest&, std::unique_ptr<NetworkCache::Entry>&&, std::optional<NavigatingToAppBoundDomain>, bool allowPrivacyProxy, OptionSet<WebCore::AdvancedPrivacyProtections>, CompletionHandler<void(Result)>&&);

    void cancel();

    const SpeculativeLoad& load() const { return *m_load; }

private:
    AsyncRevalidation(Cache&, const GlobalFrameID&, const WebCore::ResourceRequest&, std::unique_ptr<NetworkCache::Entry>&&, std::optional<NavigatingToAppBoundDomain>, bool allowPrivacyProxy, OptionSet<WebCore::AdvancedPrivacyProtections>, CompletionHandler<void(Result)>&&);
    void staleWhileRevalidateEnding();

    std::unique_ptr<SpeculativeLoad> m_load;
    WebCore::Timer m_timer;
    CompletionHandler<void(Result)> m_completionHandler;
};

} // namespace NetworkCache
} // namespace WebKit

#endif // ENABLE(NETWORK_CACHE_STALE_WHILE_REVALIDATE)
