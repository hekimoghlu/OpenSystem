/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 13, 2025.
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
#include "config.h"
#include "APIPageConfiguration.h"

#include "APIProcessPoolConfiguration.h"
#include "APIWebsitePolicies.h"
#include "BrowsingContextGroup.h"
#include "Logging.h"
#include "WebInspectorUtilities.h"
#include "WebPageGroup.h"
#include "WebPageProxy.h"
#include "WebPreferences.h"
#include "WebProcessPool.h"
#include "WebURLSchemeHandler.h"
#include "WebUserContentControllerProxy.h"

#if ENABLE(APPLICATION_MANIFEST)
#include "APIApplicationManifest.h"
#endif

#if ENABLE(WK_WEB_EXTENSIONS)
#include "WebExtensionController.h"
#include "WebExtensionMatchPattern.h"
#endif

namespace API {
using namespace WebKit;

PageConfiguration::Data::Data()
    : openedSite(aboutBlankURL()) { }

Ref<WebKit::BrowsingContextGroup> PageConfiguration::Data::createBrowsingContextGroup()
{
    return BrowsingContextGroup::create();
}

Ref<WebKit::WebProcessPool> PageConfiguration::Data::createWebProcessPool()
{
    return WebProcessPool::create(ProcessPoolConfiguration::create());
}

Ref<WebKit::WebUserContentControllerProxy> PageConfiguration::Data::createWebUserContentControllerProxy()
{
    return WebUserContentControllerProxy::create();
}

Ref<WebKit::WebPreferences> PageConfiguration::Data::createWebPreferences()
{
    return WebPreferences::create(WTF::String(), "WebKit"_s, "WebKitDebug"_s);
}

Ref<WebKit::VisitedLinkStore> PageConfiguration::Data::createVisitedLinkStore()
{
    return WebKit::VisitedLinkStore::create();
}

Ref<WebsitePolicies> PageConfiguration::Data::createWebsitePolicies()
{
    return WebsitePolicies::create();
}

Ref<PageConfiguration> PageConfiguration::create()
{
    return adoptRef(*new PageConfiguration);
}

PageConfiguration::PageConfiguration() = default;

PageConfiguration::~PageConfiguration() = default;

Ref<PageConfiguration> PageConfiguration::copy() const
{
    auto copy = create();
    copy->m_data = m_data;
    return copy;
}

void PageConfiguration::copyDataFrom(const PageConfiguration& other)
{
    m_data = other.m_data;
}

BrowsingContextGroup& PageConfiguration::browsingContextGroup() const
{
    return m_data.browsingContextGroup.get();
}

void PageConfiguration::setBrowsingContextGroup(RefPtr<BrowsingContextGroup>&& group)
{
    m_data.browsingContextGroup = WTFMove(group);
}

const std::optional<WebCore::WindowFeatures>& PageConfiguration::windowFeatures() const
{
    return m_data.windowFeatures;
}

void PageConfiguration::setWindowFeatures(WebCore::WindowFeatures&& windowFeatures)
{
    m_data.windowFeatures = WTFMove(windowFeatures);
}

const WebCore::Site& PageConfiguration::openedSite() const
{
    return m_data.openedSite;
}

void PageConfiguration::setOpenedSite(const WebCore::Site& site)
{
    m_data.openedSite = site;
}

const WTF::String& PageConfiguration::openedMainFrameName() const
{
    return m_data.openedMainFrameName;
}

void PageConfiguration::setOpenedMainFrameName(const WTF::String& name)
{
    m_data.openedMainFrameName = name;
}

auto PageConfiguration::openerInfo() const -> const std::optional<OpenerInfo>&
{
    return m_data.openerInfo;
}

void PageConfiguration::setOpenerInfo(std::optional<OpenerInfo>&& info)
{
    m_data.openerInfo = WTFMove(info);
}

bool PageConfiguration::OpenerInfo::operator==(const OpenerInfo&) const = default;

void PageConfiguration::setInitialSandboxFlags(WebCore::SandboxFlags sandboxFlags)
{
    m_data.initialSandboxFlags = sandboxFlags;
}

WebProcessPool& PageConfiguration::processPool() const
{
    return m_data.processPool.get();
}

Ref<WebKit::WebProcessPool> PageConfiguration::protectedProcessPool() const
{
    return processPool();
}

void PageConfiguration::setProcessPool(RefPtr<WebProcessPool>&& processPool)
{
    m_data.processPool = WTFMove(processPool);
}

WebUserContentControllerProxy& PageConfiguration::userContentController() const
{
    return m_data.userContentController.get();
}

void PageConfiguration::setUserContentController(RefPtr<WebUserContentControllerProxy>&& userContentController)
{
    m_data.userContentController = WTFMove(userContentController);
}

#if ENABLE(WK_WEB_EXTENSIONS)
const WTF::URL& PageConfiguration::requiredWebExtensionBaseURL() const
{
    return m_data.requiredWebExtensionBaseURL;
}

void PageConfiguration::setRequiredWebExtensionBaseURL(WTF::URL&& baseURL)
{
    m_data.requiredWebExtensionBaseURL = WTFMove(baseURL);
}

WebExtensionController* PageConfiguration::webExtensionController() const
{
    return m_data.webExtensionController.get();
}

void PageConfiguration::setWebExtensionController(RefPtr<WebExtensionController>&& webExtensionController)
{
    m_data.webExtensionController = WTFMove(webExtensionController);
}

WebExtensionController* PageConfiguration::weakWebExtensionController() const
{
    return m_data.weakWebExtensionController.get();
}

void PageConfiguration::setWeakWebExtensionController(WebExtensionController* webExtensionController)
{
    m_data.weakWebExtensionController = webExtensionController;
}
#endif // ENABLE(WK_WEB_EXTENSIONS)


HashSet<WTF::String> PageConfiguration::maskedURLSchemes() const
{
    if (m_data.maskedURLSchemesWasSet)
        return m_data.maskedURLSchemes;
#if ENABLE(WK_WEB_EXTENSIONS) && PLATFORM(COCOA)
    if (webExtensionController() || weakWebExtensionController())
        return WebKit::WebExtensionMatchPattern::extensionSchemes();
#endif
    return { };
}

WebPageGroup* PageConfiguration::pageGroup()
{
    return m_data.pageGroup.get();
}

void PageConfiguration::setPageGroup(RefPtr<WebPageGroup>&& pageGroup)
{
    m_data.pageGroup = WTFMove(pageGroup);
}

WebPreferences& PageConfiguration::preferences() const
{
    return m_data.preferences.get();
}

void PageConfiguration::setPreferences(RefPtr<WebPreferences>&& preferences)
{
    m_data.preferences = WTFMove(preferences);
}

WebPageProxy* PageConfiguration::relatedPage() const
{
    return m_data.relatedPage.get();
}

WebPageProxy* PageConfiguration::pageToCloneSessionStorageFrom() const
{
    return m_data.pageToCloneSessionStorageFrom.get();
}

void PageConfiguration::setPageToCloneSessionStorageFrom(WeakPtr<WebPageProxy>&& pageToCloneSessionStorageFrom)
{
    m_data.pageToCloneSessionStorageFrom = WTFMove(pageToCloneSessionStorageFrom);
}

WebPageProxy* PageConfiguration::alternateWebViewForNavigationGestures() const
{
    return m_data.alternateWebViewForNavigationGestures.get();
}

void PageConfiguration::setAlternateWebViewForNavigationGestures(WeakPtr<WebPageProxy>&& alternateWebViewForNavigationGestures)
{
    m_data.alternateWebViewForNavigationGestures = WTFMove(alternateWebViewForNavigationGestures);
}

WebKit::VisitedLinkStore& PageConfiguration::visitedLinkStore() const
{
    return m_data.visitedLinkStore.get();
}

void PageConfiguration::setVisitedLinkStore(RefPtr<WebKit::VisitedLinkStore>&& visitedLinkStore)
{
    m_data.visitedLinkStore = WTFMove(visitedLinkStore);
}

WebsiteDataStore& PageConfiguration::websiteDataStore() const
{
    if (!m_data.websiteDataStore)
        m_data.websiteDataStore = WebsiteDataStore::defaultDataStore();
    return *m_data.websiteDataStore;
}

WebKit::WebsiteDataStore* PageConfiguration::websiteDataStoreIfExists() const
{
    return m_data.websiteDataStore.get();
}

Ref<WebsiteDataStore> PageConfiguration::protectedWebsiteDataStore() const
{
    return websiteDataStore();
}

void PageConfiguration::setWebsiteDataStore(RefPtr<WebsiteDataStore>&& websiteDataStore)
{
    m_data.websiteDataStore = WTFMove(websiteDataStore);
}

WebsitePolicies& PageConfiguration::defaultWebsitePolicies() const
{
    return m_data.defaultWebsitePolicies.get();
}

void PageConfiguration::setDefaultWebsitePolicies(RefPtr<WebsitePolicies>&& policies)
{
    m_data.defaultWebsitePolicies = WTFMove(policies);
}

RefPtr<WebURLSchemeHandler> PageConfiguration::urlSchemeHandlerForURLScheme(const WTF::String& scheme)
{
    return m_data.urlSchemeHandlers.get(scheme);
}

void PageConfiguration::setURLSchemeHandlerForURLScheme(Ref<WebURLSchemeHandler>&& handler, const WTF::String& scheme)
{
    m_data.urlSchemeHandlers.set(scheme, WTFMove(handler));
}

bool PageConfiguration::lockdownModeEnabled() const
{
    if (RefPtr policies = m_data.defaultWebsitePolicies.getIfExists())
        return policies->lockdownModeEnabled();
    return lockdownModeEnabledBySystem();
}

void PageConfiguration::setDelaysWebProcessLaunchUntilFirstLoad(bool delaysWebProcessLaunchUntilFirstLoad)
{
    RELEASE_LOG(Process, "%p - PageConfiguration::setDelaysWebProcessLaunchUntilFirstLoad(%d)", this, delaysWebProcessLaunchUntilFirstLoad);
    m_data.delaysWebProcessLaunchUntilFirstLoad = delaysWebProcessLaunchUntilFirstLoad;
}

bool PageConfiguration::delaysWebProcessLaunchUntilFirstLoad() const
{
    if (preferences().siteIsolationEnabled())
        return true;
    if (RefPtr processPool = m_data.processPool.getIfExists(); processPool && isInspectorProcessPool(*processPool)) {
        // Never delay process launch for inspector pages as inspector pages do not know how to transition from a terminated process.
        RELEASE_LOG(Process, "%p - PageConfiguration::delaysWebProcessLaunchUntilFirstLoad() -> false because of WebInspector pool", this);
        return false;
    }
    if (m_data.delaysWebProcessLaunchUntilFirstLoad) {
        RELEASE_LOG(Process, "%p - PageConfiguration::delaysWebProcessLaunchUntilFirstLoad() -> %{public}s because of explicit client value", this, *m_data.delaysWebProcessLaunchUntilFirstLoad ? "true" : "false");
        // If the client explicitly enabled / disabled the feature, then obey their directives.
        return *m_data.delaysWebProcessLaunchUntilFirstLoad;
    }
    if (RefPtr processPool = m_data.processPool.getIfExists()) {
        RELEASE_LOG(Process, "%p - PageConfiguration::delaysWebProcessLaunchUntilFirstLoad() -> %{public}s because of associated processPool value", this, processPool->delaysWebProcessLaunchDefaultValue() ? "true" : "false");
        return processPool->delaysWebProcessLaunchDefaultValue();
    }
    RELEASE_LOG(Process, "%p - PageConfiguration::delaysWebProcessLaunchUntilFirstLoad() -> %{public}s because of global default value", this, WebProcessPool::globalDelaysWebProcessLaunchDefaultValue() ? "true" : "false");
    return WebProcessPool::globalDelaysWebProcessLaunchDefaultValue();
}

bool PageConfiguration::isLockdownModeExplicitlySet() const
{
    if (RefPtr policies = m_data.defaultWebsitePolicies.getIfExists())
        return policies->isLockdownModeExplicitlySet();
    return false;
}

#if ENABLE(APPLICATION_MANIFEST)
ApplicationManifest* PageConfiguration::applicationManifest() const
{
    return m_data.applicationManifest.get();
}

void PageConfiguration::setApplicationManifest(RefPtr<ApplicationManifest>&& applicationManifest)
{
    m_data.applicationManifest = WTFMove(applicationManifest);
}
#endif

#if ENABLE(APPLE_PAY)

bool PageConfiguration::applePayEnabled() const
{
    if (auto applePayEnabledOverride = m_data.applePayEnabledOverride)
        return *applePayEnabledOverride;

    return preferences().applePayEnabled();
}

void PageConfiguration::setApplePayEnabled(bool enabled)
{
    m_data.applePayEnabledOverride = enabled;
}

#endif

} // namespace API
