/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 2, 2022.
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

#include "ResourceLoadStatisticsStore.h"
#include "WebResourceLoadStatisticsStore.h"
#include <wtf/CompletionHandler.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class KeyedDecoder;
class KeyedEncoder;
enum class StorageAccessPromptWasShown : bool;
enum class StorageAccessWasGranted : bool;
struct ResourceLoadStatistics;
}

namespace WebKit {

// This is always constructed / used / destroyed on the WebResourceLoadStatisticsStore's statistics queue.
class ResourceLoadStatisticsMemoryStore final : public ResourceLoadStatisticsStore {
public:
    ResourceLoadStatisticsMemoryStore(WebResourceLoadStatisticsStore&, SuspendableWorkQueue&, ShouldIncludeLocalhost);

    void clear(CompletionHandler<void()>&&) override;
    bool isEmpty() const override;

    Vector<ITPThirdPartyData> aggregatedThirdPartyData() const override;
    const HashMap<RegistrableDomain, UniqueRef<WebCore::ResourceLoadStatistics>>& data() const { return m_resourceStatisticsMap; }

    std::unique_ptr<WebCore::KeyedEncoder> createEncoderFromData() const;
    void mergeWithDataFromDecoder(WebCore::KeyedDecoder&);

    void mergeStatistics(Vector<ResourceLoadStatistics>&&) override;
    void processStatistics(const Function<void(const ResourceLoadStatistics&)>&) const;

    void updateCookieBlocking(CompletionHandler<void()>&&) override;

    void classifyPrevalentResources() override;

    void requestStorageAccessUnderOpener(DomainInNeedOfStorageAccess&&, WebCore::PageIdentifier, OpenerDomain&&) override;

    void grandfatherDataForDomains(const HashSet<RegistrableDomain>&) override;

    bool isRegisteredAsSubresourceUnder(const SubResourceDomain&, const TopFrameDomain&) const override;
    bool isRegisteredAsSubFrameUnder(const SubFrameDomain&, const TopFrameDomain&) const override;
    bool isRegisteredAsRedirectingTo(const RedirectedFromDomain&, const RedirectedToDomain&) const override;

    void clearPrevalentResource(const RegistrableDomain&) override;
    void dumpResourceLoadStatistics(CompletionHandler<void(String&&)>&&) final;
    bool isPrevalentResource(const RegistrableDomain&) const override;
    bool isVeryPrevalentResource(const RegistrableDomain&) const override;
    void setPrevalentResource(const RegistrableDomain&) override;
    void setVeryPrevalentResource(const RegistrableDomain&) override;

    void setGrandfathered(const RegistrableDomain&, bool value) override;
    bool isGrandfathered(const RegistrableDomain&) const override;

    void setSubframeUnderTopFrameDomain(const SubFrameDomain&, const TopFrameDomain&) override;
    void setSubresourceUnderTopFrameDomain(const SubResourceDomain&, const TopFrameDomain&) override;
    void setSubresourceUniqueRedirectTo(const SubResourceDomain&, const RedirectDomain&) override;
    void setSubresourceUniqueRedirectFrom(const SubResourceDomain&, const RedirectDomain&) override;
    void setTopFrameUniqueRedirectTo(const TopFrameDomain&, const RedirectDomain&) override;
    void setTopFrameUniqueRedirectFrom(const TopFrameDomain&, const RedirectDomain&) override;

    bool areAllThirdPartyCookiesBlockedUnder(const TopFrameDomain&) override;
    CookieAccess cookieAccess(const ResourceLoadStatistics&, const TopFrameDomain&);
    void hasStorageAccess(SubFrameDomain&&, TopFrameDomain&&, std::optional<WebCore::FrameIdentifier>, WebCore::PageIdentifier, CompletionHandler<void(bool)>&&) override;
    void requestStorageAccess(SubFrameDomain&&, TopFrameDomain&&, WebCore::FrameIdentifier, WebCore::PageIdentifier, WebCore::StorageAccessScope, CompletionHandler<void(StorageAccessStatus)>&&) override;
    void grantStorageAccess(SubFrameDomain&&, TopFrameDomain&&, WebCore::FrameIdentifier, WebCore::PageIdentifier, WebCore::StorageAccessPromptWasShown, WebCore::StorageAccessScope, CompletionHandler<void(WebCore::StorageAccessWasGranted)>&&) override;

    void logFrameNavigation(const NavigatedToDomain&, const TopFrameDomain&, const NavigatedFromDomain&, bool isRedirect, bool isMainFrame, Seconds delayAfterMainFrameDocumentLoad, bool wasPotentiallyInitiatedByUser) override;
    void logUserInteraction(const TopFrameDomain&, CompletionHandler<void()>&&) override;
    void logCrossSiteLoadWithLinkDecoration(const NavigatedFromDomain&, const NavigatedToDomain&) override;

    void clearUserInteraction(const RegistrableDomain&, CompletionHandler<void()>&&) override;
    bool hasHadUserInteraction(const RegistrableDomain&, OperatingDatesWindow) override;

    void setLastSeen(const RegistrableDomain&, Seconds) override;
    void removeDataForDomain(const RegistrableDomain&) override;
    Vector<RegistrableDomain> allDomains() const final;
    void insertExpiredStatisticForTesting(const RegistrableDomain&, unsigned numberOfOperatingDaysPassed, bool hasUserInteraction, bool isScheduledForAllButCookieDataRemoval, bool isPrevalent) override;

private:
    void includeTodayAsOperatingDateIfNecessary() override;
    const Vector<OperatingDate>& operatingDates() const { return m_operatingDates; }
    void clearOperatingDates() override { m_operatingDates.clear(); }
    void mergeOperatingDates(Vector<OperatingDate>&&);
    bool hasStatisticsExpired(const ResourceLoadStatistics&, OperatingDatesWindow) const;
    static Vector<OperatingDate> mergeOperatingDates(const Vector<OperatingDate>& existingDates, Vector<OperatingDate>&& newDates);
    bool hasStatisticsExpired(WallTime mostRecentUserInteractionTime, OperatingDatesWindow) const override;

    static bool shouldBlockAndKeepCookies(const ResourceLoadStatistics&);
    static bool shouldBlockAndPurgeCookies(const ResourceLoadStatistics&);
    static WebCore::StorageAccessPromptWasShown hasUserGrantedStorageAccessThroughPrompt(const ResourceLoadStatistics&, const RegistrableDomain&);
    bool hasHadUnexpiredRecentUserInteraction(ResourceLoadStatistics&, OperatingDatesWindow) const;
    bool shouldRemoveAllWebsiteDataFor(ResourceLoadStatistics&, bool shouldCheckForGrandfathering) const;
    bool shouldRemoveAllButCookiesFor(ResourceLoadStatistics&, bool shouldCheckForGrandfathering) const;
    bool shouldEnforceSameSiteStrictFor(ResourceLoadStatistics&, bool shouldCheckForGrandfathering);
    void incrementRecordsDeletedCountForDomains(HashSet<RegistrableDomain>&&) override;
    void setPrevalentResource(ResourceLoadStatistics&, ResourceLoadPrevalence);
    unsigned recursivelyGetAllDomainsThatHaveRedirectedToThisDomain(const ResourceLoadStatistics&, HashSet<RedirectedToDomain>&, unsigned numberOfRecursiveCalls) const;
    void grantStorageAccessInternal(SubFrameDomain&&, TopFrameDomain&&, std::optional<WebCore::FrameIdentifier>, WebCore::PageIdentifier, WebCore::StorageAccessPromptWasShown, WebCore::StorageAccessScope, CompletionHandler<void(WebCore::StorageAccessWasGranted)>&&);
    void markAsPrevalentIfHasRedirectedToPrevalent(ResourceLoadStatistics&);
    bool isPrevalentDueToDebugMode(ResourceLoadStatistics&);
    Vector<RegistrableDomain> ensurePrevalentResourcesForDebugMode() override;
    void removeDataRecords(CompletionHandler<void()>&&);
    void pruneStatisticsIfNeeded() override;
    ResourceLoadStatistics& ensureResourceStatisticsForRegistrableDomain(const RegistrableDomain&);
    RegistrableDomainsToDeleteOrRestrictWebsiteDataFor registrableDomainsToDeleteOrRestrictWebsiteDataFor() override;
    bool isMemoryStore() const final { return true; }
    std::optional<WallTime> mostRecentUserInteractionTime(const ResourceLoadStatistics&);

    HashMap<RegistrableDomain, UniqueRef<ResourceLoadStatistics>> m_resourceStatisticsMap;
    Vector<OperatingDate> m_operatingDates;
};

} // namespace WebKit

SPECIALIZE_TYPE_TRAITS_BEGIN(WebKit::ResourceLoadStatisticsMemoryStore)
    static bool isType(const WebKit::ResourceLoadStatisticsStore& store) { return store.isMemoryStore(); }
SPECIALIZE_TYPE_TRAITS_END()
