/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 30, 2021.
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
#ifndef NetworkCacheSpeculativeLoadManager_h
#define NetworkCacheSpeculativeLoadManager_h

#if ENABLE(NETWORK_CACHE_SPECULATIVE_REVALIDATION)

#include "NetworkCache.h"
#include "NetworkCacheStorage.h"
#include <WebCore/ResourceRequest.h>
#include <wtf/CheckedPtr.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

namespace WebKit {
namespace NetworkCache {
class SpeculativeLoadManager;
}
}

namespace WebCore {
enum class AdvancedPrivacyProtections : uint16_t;
}

namespace WebKit {

namespace NetworkCache {

class Entry;
class SpeculativeLoad;
class SubresourceInfo;
class SubresourcesEntry;

class SpeculativeLoadManager final : public CanMakeWeakPtr<SpeculativeLoadManager>, public CanMakeCheckedPtr<SpeculativeLoadManager> {
    WTF_MAKE_TZONE_ALLOCATED(SpeculativeLoadManager);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SpeculativeLoadManager);
public:
    explicit SpeculativeLoadManager(Cache&, Storage&);
    ~SpeculativeLoadManager();

    void registerLoad(GlobalFrameID, const WebCore::ResourceRequest&, const Key& resourceKey, std::optional<NavigatingToAppBoundDomain>, bool allowPrivacyProxy, OptionSet<WebCore::AdvancedPrivacyProtections>);
    void registerMainResourceLoadResponse(const GlobalFrameID&, const WebCore::ResourceRequest&, const WebCore::ResourceResponse&);

    typedef Function<void (std::unique_ptr<Entry>)> RetrieveCompletionHandler;

    bool canRetrieve(const Key& storageKey, const WebCore::ResourceRequest&, const GlobalFrameID&) const;
    void retrieve(const Key& storageKey, RetrieveCompletionHandler&&);

private:
    class PreloadedEntry;

    static bool shouldRegisterLoad(const WebCore::ResourceRequest&);
    void addPreloadedEntry(std::unique_ptr<Entry>, const GlobalFrameID&, std::optional<WebCore::ResourceRequest>&& revalidationRequest = std::nullopt);
    void preloadEntry(const Key&, const SubresourceInfo&, const GlobalFrameID&, std::optional<NavigatingToAppBoundDomain>, bool allowPrivacyProxy, OptionSet<WebCore::AdvancedPrivacyProtections>);
    void retrieveEntryFromStorage(const SubresourceInfo&, RetrieveCompletionHandler&&);
    void revalidateSubresource(const SubresourceInfo&, std::unique_ptr<Entry>, const GlobalFrameID&, std::optional<NavigatingToAppBoundDomain>, bool allowPrivacyProxy, OptionSet<WebCore::AdvancedPrivacyProtections>);
    void preconnectForSubresource(const SubresourceInfo&, Entry*, const GlobalFrameID&, std::optional<NavigatingToAppBoundDomain>);
    bool satisfyPendingRequests(const Key&, Entry*);
    void retrieveSubresourcesEntry(const Key& storageKey, WTF::Function<void (std::unique_ptr<SubresourcesEntry>)>&&);
    void startSpeculativeRevalidation(const GlobalFrameID&, SubresourcesEntry&, bool, std::optional<NavigatingToAppBoundDomain>, bool allowPrivacyProxy, OptionSet<WebCore::AdvancedPrivacyProtections>);

    static bool canUsePreloadedEntry(const PreloadedEntry&, const WebCore::ResourceRequest& actualRequest);
    static bool canUsePendingPreload(const SpeculativeLoad&, const WebCore::ResourceRequest& actualRequest);

    Ref<Storage> protectedStorage() const;

    WeakRef<Cache> m_cache;
    ThreadSafeWeakPtr<Storage> m_storage; // Not expected to be null.

    class PendingFrameLoad;
    HashMap<GlobalFrameID, RefPtr<PendingFrameLoad>> m_pendingFrameLoads;

    HashMap<Key, std::unique_ptr<SpeculativeLoad>> m_pendingPreloads;
    HashMap<Key, std::unique_ptr<Vector<RetrieveCompletionHandler>>> m_pendingRetrieveRequests;

    HashMap<Key, std::unique_ptr<PreloadedEntry>> m_preloadedEntries;

    class ExpiringEntry;
    HashMap<Key, std::unique_ptr<ExpiringEntry>> m_notPreloadedEntries; // For logging.
};

} // namespace NetworkCache

} // namespace WebKit

#endif // ENABLE(NETWORK_CACHE_SPECULATIVE_REVALIDATION)

#endif // NetworkCacheSpeculativeLoadManager_h
