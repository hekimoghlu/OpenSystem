/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 22, 2024.
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

#include "DebuggableInfoData.h"
#include "MessageReceiver.h"
#include <WebCore/Color.h>
#include <WebCore/FrameIdentifier.h>
#include <WebCore/InspectorFrontendAPIDispatcher.h>
#include <WebCore/InspectorFrontendClient.h>
#include <WebCore/InspectorFrontendHost.h>
#include <wtf/Deque.h>
#include <wtf/WeakRef.h>

#if ENABLE(INSPECTOR_EXTENSIONS)
#include "InspectorExtensionTypes.h"
#endif

namespace WebCore {
class CertificateInfo;
class FloatRect;
}

namespace WebKit {

class WebPage;
#if ENABLE(INSPECTOR_EXTENSIONS)
class WebInspectorUIExtensionController;
#endif

class RemoteWebInspectorUI final
    : public RefCounted<RemoteWebInspectorUI>
    , public IPC::MessageReceiver
    , public WebCore::InspectorFrontendClient {
public:
    static Ref<RemoteWebInspectorUI> create(WebPage&);
    ~RemoteWebInspectorUI();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    // Implemented in generated RemoteWebInspectorUIMessageReceiver.cpp
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

    // Called by RemoteWebInspectorUI messages
    void initialize(DebuggableInfoData&&, const String& backendCommandsURL);
    void updateFindString(const String&);
    void sendMessageToFrontend(const String&);
    void showConsole();
    void showResources();

#if ENABLE(INSPECTOR_TELEMETRY)
    void setDiagnosticLoggingAvailable(bool);
#endif

    // WebCore::InspectorFrontendClient
    void windowObjectCleared() override;
    void frontendLoaded() override;

    void pagePaused() override;
    void pageUnpaused() override;

    void changeSheetRect(const WebCore::FloatRect&) override;
    void startWindowDrag() override;
    void moveWindowBy(float x, float y) override;

    bool isRemote() const final { return true; }
    String localizedStringsURL() const override;
    String backendCommandsURL() const final { return m_backendCommandsURL; }
    Inspector::DebuggableType debuggableType() const override;
    String targetPlatformName() const override;
    String targetBuildVersion() const override;
    String targetProductVersion() const override;
    bool targetIsSimulator() const override;

    void setForcedAppearance(WebCore::InspectorFrontendClient::Appearance) override;

    WebCore::UserInterfaceLayoutDirection userInterfaceLayoutDirection() const override;

    bool supportsDockSide(DockSide) override;

    void bringToFront() override;
    void closeWindow() override;
    void reopen() override;
    void resetState() override;

    void openURLExternally(const String& url) override;
    void revealFileExternally(const String& path) override;
    void save(Vector<WebCore::InspectorFrontendClient::SaveData>&&, bool forceSaveAs) override;
    void load(const String& path, CompletionHandler<void(const String&)>&&) override;
    void pickColorFromScreen(CompletionHandler<void(const std::optional<WebCore::Color>&)>&&) override;
    void inspectedURLChanged(const String&) override;
    void showCertificate(const WebCore::CertificateInfo&) override;
    void setInspectorPageDeveloperExtrasEnabled(bool) override;
    void sendMessageToBackend(const String&) override;
    WebCore::InspectorFrontendAPIDispatcher& frontendAPIDispatcher() override { return m_frontendAPIDispatcher; }
    WebCore::Page* frontendPage() final;

#if ENABLE(INSPECTOR_TELEMETRY)
    bool supportsDiagnosticLogging() override;
    bool diagnosticLoggingAvailable() override { return m_diagnosticLoggingAvailable; }
    void logDiagnosticEvent(const String& eventName, const WebCore::DiagnosticLoggingClient::ValueDictionary&) override;
#endif
        
#if ENABLE(INSPECTOR_EXTENSIONS)
    bool supportsWebExtensions() override;
    void didShowExtensionTab(const Inspector::ExtensionID&, const Inspector::ExtensionTabID&, const WebCore::FrameIdentifier&) override;
    void didHideExtensionTab(const Inspector::ExtensionID&, const Inspector::ExtensionTabID&) override;
    void didNavigateExtensionTab(const Inspector::ExtensionID&, const Inspector::ExtensionTabID&, const URL&) override;
    void inspectedPageDidNavigate(const URL&) override;
#endif

    bool canSave(WebCore::InspectorFrontendClient::SaveMode) override;
    bool canLoad() override;
    bool canPickColorFromScreen() override;
    bool isUnderTest() override { return false; }
    unsigned inspectionLevel() const override { return 1; }
    void requestSetDockSide(DockSide) override { }
    void changeAttachedWindowHeight(unsigned) override { }
    void changeAttachedWindowWidth(unsigned) override { }

private:
    explicit RemoteWebInspectorUI(WebPage&);

    WeakRef<WebPage> m_page;
    Ref<WebCore::InspectorFrontendAPIDispatcher> m_frontendAPIDispatcher;
    RefPtr<WebCore::InspectorFrontendHost> m_frontendHost;
#if ENABLE(INSPECTOR_EXTENSIONS)
    RefPtr<WebInspectorUIExtensionController> m_extensionController;
#endif

    DebuggableInfoData m_debuggableInfo;
    String m_backendCommandsURL;

#if ENABLE(INSPECTOR_TELEMETRY)
    bool m_diagnosticLoggingAvailable { false };
#endif
};

} // namespace WebKit
