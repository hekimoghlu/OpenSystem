/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 21, 2024.
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

#include <WebCore/PageIdentifier.h>
#include <WebCore/ResourceLoadObserver.h>
#include <WebCore/ResourceLoadStatistics.h>
#include <WebCore/Timer.h>
#include <wtf/Forward.h>

namespace WebKit {

class WebPage;

class WebResourceLoadObserver final : public WebCore::ResourceLoadObserver {
public:
    using TopFrameDomain = WebCore::RegistrableDomain;
    using SubFrameDomain = WebCore::RegistrableDomain;

    WebResourceLoadObserver(WebCore::ResourceLoadStatistics::IsEphemeral);
    ~WebResourceLoadObserver();

    void logSubresourceLoading(const WebCore::LocalFrame*, const WebCore::ResourceRequest& newRequest, const WebCore::ResourceResponse& redirectResponse, FetchDestinationIsScriptLike) final;
    void logWebSocketLoading(const URL& targetURL, const URL& mainFrameURL) final;
    void logUserInteractionWithReducedTimeResolution(const WebCore::Document&) final;
    void logFontLoad(const WebCore::Document&, const String& familyName, bool loadStatus) final;
    void logCanvasRead(const WebCore::Document&) final;
    void logCanvasWriteOrMeasure(const WebCore::Document&, const String& textWritten) final;
    void logNavigatorAPIAccessed(const WebCore::Document&, const WebCore::NavigatorAPIsAccessed) final;
    void logScreenAPIAccessed(const WebCore::Document&, const WebCore::ScreenAPIsAccessed) final;
    void logSubresourceLoadingForTesting(const WebCore::RegistrableDomain& firstPartyDomain, const WebCore::RegistrableDomain& thirdPartyDomain, bool shouldScheduleNotification);

#if !RELEASE_LOG_DISABLED
    static void setShouldLogUserInteraction(bool);
#endif

    String statisticsForURL(const URL&) final;
    void updateCentralStatisticsStore(CompletionHandler<void()>&&) final;
    void clearState() final;
    
    bool hasStatistics() const final { return !m_resourceStatisticsMap.isEmpty(); }

    void setDomainsWithUserInteraction(HashSet<WebCore::RegistrableDomain>&& domains) final { m_domainsWithUserInteraction = WTFMove(domains); }
    void setDomainsWithCrossPageStorageAccess(HashMap<TopFrameDomain, Vector<SubFrameDomain>>&&, CompletionHandler<void()>&&) final;
    bool hasHadUserInteraction(const WebCore::RegistrableDomain&) const final;
    bool hasCrossPageStorageAccess(const SubFrameDomain&, const TopFrameDomain&) const final;

private:
    WebCore::ResourceLoadStatistics& ensureResourceStatisticsForRegistrableDomain(const WebCore::RegistrableDomain&);
    void scheduleNotificationIfNeeded();

    Vector<WebCore::ResourceLoadStatistics> takeStatistics();
    void requestStorageAccessUnderOpener(const WebCore::RegistrableDomain& domainInNeedOfStorageAccess, WebPage& openerPage, WebCore::Document& openerDocument);

    bool isEphemeral() const { return m_isEphemeral == WebCore::ResourceLoadStatistics::IsEphemeral::Yes; }

    WebCore::ResourceLoadStatistics::IsEphemeral m_isEphemeral { WebCore::ResourceLoadStatistics::IsEphemeral::No };

    HashMap<WebCore::RegistrableDomain, std::unique_ptr<WebCore::ResourceLoadStatistics>> m_resourceStatisticsMap;
    HashMap<WebCore::RegistrableDomain, WTF::WallTime> m_lastReportedUserInteractionMap;

    WebCore::Timer m_notificationTimer;

    HashSet<WebCore::RegistrableDomain> m_domainsWithUserInteraction;
    HashMap<TopFrameDomain, HashSet<SubFrameDomain>> m_domainsWithCrossPageStorageAccess;
#if !RELEASE_LOG_DISABLED
    uint64_t m_loggingCounter { 0 };
    static bool shouldLogUserInteraction;
#endif
};

} // namespace WebKit
