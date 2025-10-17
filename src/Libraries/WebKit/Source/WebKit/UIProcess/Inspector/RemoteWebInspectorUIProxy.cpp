/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 22, 2025.
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
#include "RemoteWebInspectorUIProxy.h"

#include "APIDebuggableInfo.h"
#include "APINavigation.h"
#include "MessageSenderInlines.h"
#include "RemoteWebInspectorUIMessages.h"
#include "RemoteWebInspectorUIProxyMessages.h"
#include "WebInspectorUIProxy.h"
#include "WebPageGroup.h"
#include "WebPageProxy.h"
#include <WebCore/CertificateInfo.h>
#include <WebCore/NotImplemented.h>
#include <wtf/TZoneMallocInlines.h>

#if ENABLE(INSPECTOR_EXTENSIONS)
#include "WebInspectorUIExtensionControllerProxy.h"
#endif

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteWebInspectorUIProxyClient);
WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteWebInspectorUIProxy);

RemoteWebInspectorUIProxy::RemoteWebInspectorUIProxy()
    : m_debuggableInfo(API::DebuggableInfo::create(DebuggableInfoData::empty()))
{
}

RemoteWebInspectorUIProxy::~RemoteWebInspectorUIProxy()
{
    ASSERT(!m_inspectorPage);
}

RefPtr<WebPageProxy> RemoteWebInspectorUIProxy::protectedInspectorPage()
{
    return m_inspectorPage.get();
}

#if ENABLE(INSPECTOR_EXTENSIONS)
RefPtr<WebInspectorUIExtensionControllerProxy> RemoteWebInspectorUIProxy::protectedExtensionController()
{
    return m_extensionController;
}
#endif

void RemoteWebInspectorUIProxy::invalidate()
{
    closeFrontendPageAndWindow();
}

void RemoteWebInspectorUIProxy::setDiagnosticLoggingAvailable(bool available)
{
#if ENABLE(INSPECTOR_TELEMETRY)
    if (RefPtr page = protectedInspectorPage())
        page->protectedLegacyMainFrameProcess()->send(Messages::RemoteWebInspectorUI::SetDiagnosticLoggingAvailable(available), page->webPageIDInMainFrameProcess());
#else
    UNUSED_PARAM(available);
#endif
}

void RemoteWebInspectorUIProxy::initialize(Ref<API::DebuggableInfo>&& debuggableInfo, const String& backendCommandsURL)
{
    m_debuggableInfo = WTFMove(debuggableInfo);
    m_backendCommandsURL = backendCommandsURL;

    createFrontendPageAndWindow();

    auto inspectorPage = protectedInspectorPage();
    inspectorPage->protectedLegacyMainFrameProcess()->send(Messages::RemoteWebInspectorUI::Initialize(m_debuggableInfo->debuggableInfoData(), backendCommandsURL), m_inspectorPage->webPageIDInMainFrameProcess());
    inspectorPage->loadRequest(URL { WebInspectorUIProxy::inspectorPageURL() });
}

void RemoteWebInspectorUIProxy::closeFromBackend()
{
    closeFrontendPageAndWindow();
}

void RemoteWebInspectorUIProxy::closeFromCrash()
{
    // Behave as if the frontend just closed, so clients are informed the frontend is gone.
    frontendDidClose();
}

void RemoteWebInspectorUIProxy::show()
{
    bringToFront();
}

void RemoteWebInspectorUIProxy::showConsole()
{
    if (RefPtr page = protectedInspectorPage())
        page->protectedLegacyMainFrameProcess()->send(Messages::RemoteWebInspectorUI::ShowConsole { }, page->webPageIDInMainFrameProcess());
}

void RemoteWebInspectorUIProxy::showResources()
{
    if (RefPtr page = protectedInspectorPage())
        page->protectedLegacyMainFrameProcess()->send(Messages::RemoteWebInspectorUI::ShowResources { }, page->webPageIDInMainFrameProcess());
}

void RemoteWebInspectorUIProxy::sendMessageToFrontend(const String& message)
{
    if (RefPtr page = protectedInspectorPage())
        page->protectedLegacyMainFrameProcess()->send(Messages::RemoteWebInspectorUI::SendMessageToFrontend(message), page->webPageIDInMainFrameProcess());
}

void RemoteWebInspectorUIProxy::frontendLoaded()
{
#if ENABLE(INSPECTOR_EXTENSIONS)
    protectedExtensionController()->inspectorFrontendLoaded();
#endif
}

void RemoteWebInspectorUIProxy::frontendDidClose()
{
    Ref<RemoteWebInspectorUIProxy> protect(*this);

    if (CheckedPtr client = m_client.get())
        client->closeFromFrontend();

    closeFrontendPageAndWindow();
}

void RemoteWebInspectorUIProxy::reopen()
{
    ASSERT(!m_backendCommandsURL.isEmpty());

    closeFrontendPageAndWindow();
    initialize(m_debuggableInfo.copyRef(), m_backendCommandsURL);
}

void RemoteWebInspectorUIProxy::resetState()
{
    platformResetState();
}

void RemoteWebInspectorUIProxy::bringToFront()
{
    platformBringToFront();
}

void RemoteWebInspectorUIProxy::save(Vector<InspectorFrontendClient::SaveData>&& saveDatas, bool forceSaveAs)
{
    platformSave(WTFMove(saveDatas), forceSaveAs);
}

void RemoteWebInspectorUIProxy::load(const String& path, CompletionHandler<void(const String&)>&& completionHandler)
{
    platformLoad(path, WTFMove(completionHandler));
}

void RemoteWebInspectorUIProxy::pickColorFromScreen(CompletionHandler<void(const std::optional<WebCore::Color>&)>&& completionHandler)
{
    platformPickColorFromScreen(WTFMove(completionHandler));
}

void RemoteWebInspectorUIProxy::setSheetRect(const FloatRect& rect)
{
    platformSetSheetRect(rect);
}

void RemoteWebInspectorUIProxy::setForcedAppearance(InspectorFrontendClient::Appearance appearance)
{
    platformSetForcedAppearance(appearance);
}

void RemoteWebInspectorUIProxy::startWindowDrag()
{
    platformStartWindowDrag();
}

void RemoteWebInspectorUIProxy::openURLExternally(const String& url)
{
    platformOpenURLExternally(url);
}

void RemoteWebInspectorUIProxy::revealFileExternally(const String& path)
{
    platformRevealFileExternally(path);
}

void RemoteWebInspectorUIProxy::showCertificate(const CertificateInfo& certificateInfo)
{
    platformShowCertificate(certificateInfo);
}

void RemoteWebInspectorUIProxy::setInspectorPageDeveloperExtrasEnabled(bool enabled)
{
    RefPtr inspectorPage = m_inspectorPage.get();
    if (!inspectorPage)
        return;

    inspectorPage->protectedPreferences()->setDeveloperExtrasEnabled(enabled);
}

void RemoteWebInspectorUIProxy::sendMessageToBackend(const String& message)
{
    if (CheckedPtr client = m_client.get())
        client->sendMessageToBackend(message);
}

void RemoteWebInspectorUIProxy::createFrontendPageAndWindow()
{
    if (m_inspectorPage)
        return;

    m_inspectorPage = platformCreateFrontendPageAndWindow();
    RefPtr inspectorPage = m_inspectorPage.get();

    trackInspectorPage(inspectorPage.get(), nullptr);

    inspectorPage->protectedLegacyMainFrameProcess()->addMessageReceiver(Messages::RemoteWebInspectorUIProxy::messageReceiverName(), inspectorPage->webPageIDInMainFrameProcess(), *this);

#if ENABLE(INSPECTOR_EXTENSIONS)
    m_extensionController = WebInspectorUIExtensionControllerProxy::create(*inspectorPage);
#endif
}

void RemoteWebInspectorUIProxy::closeFrontendPageAndWindow()
{
    RefPtr inspectorPage = protectedInspectorPage();
    if (!inspectorPage)
        return;

    inspectorPage->protectedLegacyMainFrameProcess()->removeMessageReceiver(Messages::RemoteWebInspectorUIProxy::messageReceiverName(), inspectorPage->webPageIDInMainFrameProcess());

    untrackInspectorPage(inspectorPage.get());

#if ENABLE(INSPECTOR_EXTENSIONS)
    // This extension controller may be kept alive by the IPC dispatcher beyond the point
    // when m_inspectorPage is cleared below. Notify the controller so it can clean up before then.
    protectedExtensionController()->inspectorFrontendWillClose();
    m_extensionController = nullptr;
#endif

    m_inspectorPage = nullptr;

    platformCloseFrontendPageAndWindow();
}

#if !ENABLE(REMOTE_INSPECTOR) || (!PLATFORM(MAC) && !PLATFORM(GTK) && !PLATFORM(WIN))
WebPageProxy* RemoteWebInspectorUIProxy::platformCreateFrontendPageAndWindow()
{
    notImplemented();
    return nullptr;
}

void RemoteWebInspectorUIProxy::platformResetState() { }
void RemoteWebInspectorUIProxy::platformBringToFront() { }
void RemoteWebInspectorUIProxy::platformSave(Vector<WebCore::InspectorFrontendClient::SaveData>&&, bool /* forceSaveAs */) { }
void RemoteWebInspectorUIProxy::platformLoad(const String&, CompletionHandler<void(const String&)>&& completionHandler) { completionHandler(nullString()); }
void RemoteWebInspectorUIProxy::platformPickColorFromScreen(CompletionHandler<void(const std::optional<WebCore::Color>&)>&& completionHandler) { completionHandler({ }); }
void RemoteWebInspectorUIProxy::platformSetSheetRect(const FloatRect&) { }
void RemoteWebInspectorUIProxy::platformSetForcedAppearance(InspectorFrontendClient::Appearance) { }
void RemoteWebInspectorUIProxy::platformStartWindowDrag() { }
void RemoteWebInspectorUIProxy::platformOpenURLExternally(const String&) { }
void RemoteWebInspectorUIProxy::platformRevealFileExternally(const String&) { }
void RemoteWebInspectorUIProxy::platformShowCertificate(const CertificateInfo&) { }
void RemoteWebInspectorUIProxy::platformCloseFrontendPageAndWindow() { }
#endif // !ENABLE(REMOTE_INSPECTOR) || (!PLATFORM(MAC) && !PLATFORM(GTK) && !PLATFORM(WIN))

} // namespace WebKit
