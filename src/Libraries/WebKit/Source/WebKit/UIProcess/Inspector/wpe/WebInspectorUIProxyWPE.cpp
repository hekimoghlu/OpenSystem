/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 27, 2024.
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
#include "WebInspectorUIProxy.h"

#if ENABLE(WPE_PLATFORM)

#include "APINavigationAction.h"
#include "APINavigationClient.h"
#include "APIPageConfiguration.h"
#include "WPEWebViewPlatform.h"
#include "WebFramePolicyListenerProxy.h"
#include "WebPageGroup.h"
#include "WebPageProxy.h"
#include "WebPreferences.h"
#include "WebProcessPool.h"
#include "WebsiteDataStore.h"
#include <WebCore/CertificateInfo.h>
#include <WebCore/FloatRect.h>
#include <WebCore/InspectorFrontendClient.h>
#include <WebCore/NotImplemented.h>
#include <wpe/wpe-platform.h>
#include <wtf/FileSystem.h>
#include <wtf/glib/GRefPtr.h>
#include <wtf/glib/GUniquePtr.h>
#include <wtf/text/WTFString.h>

namespace WebKit {

class InspectorNavigationClient final : public API::NavigationClient {
public:
    explicit InspectorNavigationClient(WebInspectorUIProxy& proxy)
        : m_proxy(proxy)
    {
    }

    bool processDidTerminate(WebPageProxy&, ProcessTerminationReason reason) override
    {
        if (reason == ProcessTerminationReason::Crash)
            m_proxy.closeForCrash();
        return true;
    }

    void decidePolicyForNavigationAction(WebPageProxy&, Ref<API::NavigationAction>&& navigationAction, Ref<WebFramePolicyListenerProxy>&& listener) override
    {
        // Allow non-main frames to navigate anywhere.
        if (!navigationAction->targetFrame()->isMainFrame()) {
            listener->use();
            return;
        }

        // Allow loading of the main inspector file.
        if (WebInspectorUIProxy::isMainOrTestInspectorPage(navigationAction->request().url())) {
            listener->use();
            return;
        }

        // Prevent everything else.
        listener->ignore();

        // Try to load the request in the inspected page.
        if (RefPtr page = m_proxy.protectedInspectedPage()) {
            auto request = navigationAction->request();
            page->loadRequest(WTFMove(request));
        }
    }

private:
    WebInspectorUIProxy& m_proxy;
};

static Ref<WebsiteDataStore> inspectorWebsiteDataStore()
{
    static constexpr auto versionedDirectory = "wpewebkit-" WPE_API_VERSION G_DIR_SEPARATOR_S "WebInspector" G_DIR_SEPARATOR_S ""_s;
    String baseCacheDirectory = FileSystem::pathByAppendingComponent(FileSystem::userCacheDirectory(), versionedDirectory);
    String baseDataDirectory = FileSystem::pathByAppendingComponent(FileSystem::userDataDirectory(), versionedDirectory);

    auto configuration = WebsiteDataStoreConfiguration::createWithBaseDirectories(baseCacheDirectory, baseDataDirectory);
    return WebsiteDataStore::create(WTFMove(configuration), PAL::SessionID::generatePersistentSessionID());
}

RefPtr<WebPageProxy> WebInspectorUIProxy::platformCreateFrontendPage()
{
    auto* inspectedWPEView = m_inspectedPage->wpeView();
    if (!inspectedWPEView)
        return nullptr;

    RELEASE_ASSERT(m_inspectedPage);
    RELEASE_ASSERT(!m_inspectorView);

    auto preferences = WebPreferences::create(String(), "WebKit2."_s, "WebKit2."_s);
#if ENABLE(DEVELOPER_MODE)
    // Allow developers to inspect the Web Inspector in debug builds without changing settings.
    preferences->setDeveloperExtrasEnabled(true);
    preferences->setLogsPageMessagesToSystemConsoleEnabled(true);
#endif
    preferences->setAllowTopNavigationToDataURLs(true);
    preferences->setJavaScriptRuntimeFlags({ });
    preferences->setAcceleratedCompositingEnabled(true);
    preferences->setForceCompositingMode(true);
    preferences->setThreadedScrollingEnabled(true);
    if (m_underTest)
        preferences->setHiddenPageDOMTimerThrottlingEnabled(false);

    auto pageGroup = WebPageGroup::create(WebKit::defaultInspectorPageGroupIdentifierForPage(protectedInspectedPage().get()));
    auto websiteDataStore = inspectorWebsiteDataStore();
    auto& processPool = WebKit::defaultInspectorProcessPool(inspectionLevel());

    auto pageConfiguration = API::PageConfiguration::create();
    pageConfiguration->setProcessPool(&processPool);
    pageConfiguration->setPreferences(preferences.ptr());
    pageConfiguration->setPageGroup(pageGroup.ptr());
    pageConfiguration->setWebsiteDataStore(websiteDataStore.ptr());
    m_inspectorView = WKWPE::ViewPlatform::create(wpe_view_get_display(inspectedWPEView), *pageConfiguration.ptr());

    Ref page = m_inspectorView->page();
    page->setNavigationClient(makeUniqueRef<InspectorNavigationClient>(*this));

    auto* wpeView = m_inspectorView->wpeView();
    g_signal_connect(wpeView, "closed", G_CALLBACK(+[](WPEView* wpeView, WebInspectorUIProxy* proxy) {
        proxy->close();
    }), this);
    m_inspectorWindow = wpe_view_get_toplevel(wpeView);
    wpe_view_set_toplevel(wpeView, nullptr);
    wpe_toplevel_resize(m_inspectorWindow.get(), initialWindowWidth, initialWindowHeight);

    return page;
}

void WebInspectorUIProxy::platformCreateFrontendWindow()
{
    wpe_toplevel_resize(m_inspectorWindow.get(), initialWindowWidth, initialWindowHeight);
    wpe_view_set_toplevel(m_inspectorView->wpeView(), m_inspectorWindow.get());
}

void WebInspectorUIProxy::platformCloseFrontendPageAndWindow()
{
    if (m_inspectorView)
        g_signal_handlers_disconnect_by_data(m_inspectorView->wpeView(), this);
    m_inspectorView = nullptr;
    m_inspectorWindow = nullptr;
}

void WebInspectorUIProxy::platformDidCloseForCrash()
{
    notImplemented();
}

void WebInspectorUIProxy::platformInvalidate()
{
    if (m_inspectorView)
        g_signal_handlers_disconnect_by_data(m_inspectorView->wpeView(), this);
}

void WebInspectorUIProxy::platformResetState()
{
    notImplemented();
}

void WebInspectorUIProxy::platformBringToFront()
{
    notImplemented();
}

void WebInspectorUIProxy::platformBringInspectedPageToFront()
{
    notImplemented();
}

void WebInspectorUIProxy::platformHide()
{
    notImplemented();
}

bool WebInspectorUIProxy::platformIsFront()
{
    notImplemented();
    return false;
}

void WebInspectorUIProxy::platformSetForcedAppearance(WebCore::InspectorFrontendClient::Appearance)
{
    notImplemented();
}

void WebInspectorUIProxy::platformRevealFileExternally(const String&)
{
    notImplemented();
}

void WebInspectorUIProxy::platformInspectedURLChanged(const String& url)
{
    if (!m_inspectorWindow)
        return;

    GUniquePtr<char> title(g_strdup_printf("Web Inspector â€” %s", url.utf8().data()));
    wpe_toplevel_set_title(m_inspectorWindow.get(), title.get());
}

void WebInspectorUIProxy::platformShowCertificate(const WebCore::CertificateInfo&)
{
    notImplemented();
}

void WebInspectorUIProxy::platformSave(Vector<WebCore::InspectorFrontendClient::SaveData>&&, bool /* forceSaveAs */)
{
    notImplemented();
}

void WebInspectorUIProxy::platformLoad(const String&, CompletionHandler<void(const String&)>&& completionHandler)
{
    notImplemented();
    completionHandler(nullString());
}

void WebInspectorUIProxy::platformPickColorFromScreen(CompletionHandler<void(const std::optional<WebCore::Color>&)>&& completionHandler)
{
    notImplemented();
    completionHandler({ });
}

void WebInspectorUIProxy::platformAttach()
{
    notImplemented();
}

void WebInspectorUIProxy::platformDetach()
{
    notImplemented();
}

void WebInspectorUIProxy::platformSetAttachedWindowHeight(unsigned)
{
    notImplemented();
}

void WebInspectorUIProxy::platformSetSheetRect(const WebCore::FloatRect&)
{
    notImplemented();
}

void WebInspectorUIProxy::platformStartWindowDrag()
{
    notImplemented();
}

String WebInspectorUIProxy::inspectorPageURL()
{
    return "resource:///org/webkit/inspector/UserInterface/Main.html"_s;
}

String WebInspectorUIProxy::inspectorTestPageURL()
{
    return "resource:///org/webkit/inspector/UserInterface/Test.html"_s;
}

DebuggableInfoData WebInspectorUIProxy::infoForLocalDebuggable()
{
    auto data = DebuggableInfoData::empty();
    data.debuggableType = Inspector::DebuggableType::WebPage;
    return data;
}

void WebInspectorUIProxy::platformSetAttachedWindowWidth(unsigned)
{
    notImplemented();
}

void WebInspectorUIProxy::platformAttachAvailabilityChanged(bool)
{
    notImplemented();
}

} // namespace WebKit

#endif // ENABLE(WPE_PLATFORM)
