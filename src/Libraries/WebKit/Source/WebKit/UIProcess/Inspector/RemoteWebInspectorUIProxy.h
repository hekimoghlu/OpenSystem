/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 10, 2022.
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

#include "APIObject.h"
#include "MessageReceiver.h"
#include <WebCore/Color.h>
#include <WebCore/FloatRect.h>
#include <WebCore/InspectorFrontendClient.h>
#include <wtf/Forward.h>
#include <wtf/HashMap.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

#if PLATFORM(MAC)
OBJC_CLASS NSURL;
OBJC_CLASS NSWindow;
OBJC_CLASS WKInspectorViewController;
OBJC_CLASS WKRemoteWebInspectorUIProxyObjCAdapter;
OBJC_CLASS WKWebView;
#elif PLATFORM(GTK)
#include <wtf/glib/GWeakPtr.h>
#endif

namespace WebCore {
class CertificateInfo;
}

namespace API {
class DebuggableInfo;
class InspectorConfiguration;
}

namespace WebKit {

class RemoteWebInspectorUIProxy;
class WebPageProxy;
class WebView;
#if ENABLE(INSPECTOR_EXTENSIONS)
class WebInspectorUIExtensionControllerProxy;
#endif

class RemoteWebInspectorUIProxyClient : public CanMakeWeakPtr<RemoteWebInspectorUIProxyClient>, public CanMakeCheckedPtr<RemoteWebInspectorUIProxyClient> {
    WTF_MAKE_TZONE_ALLOCATED(RemoteWebInspectorUIProxyClient);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RemoteWebInspectorUIProxyClient);
public:
    virtual ~RemoteWebInspectorUIProxyClient() { }
    virtual void sendMessageToBackend(const String& message) = 0;
    virtual void closeFromFrontend() = 0;
    virtual Ref<API::InspectorConfiguration> configurationForRemoteInspector(RemoteWebInspectorUIProxy&) = 0;
};

class RemoteWebInspectorUIProxy : public RefCounted<RemoteWebInspectorUIProxy>, public IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(RemoteWebInspectorUIProxy);
public:
    static Ref<RemoteWebInspectorUIProxy> create()
    {
        return adoptRef(*new RemoteWebInspectorUIProxy());
    }

    ~RemoteWebInspectorUIProxy();

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    void setClient(RemoteWebInspectorUIProxyClient* client) { m_client = client; }

    bool isUnderTest() const { return false; }

    void setDiagnosticLoggingAvailable(bool);

    void invalidate();

    void initialize(Ref<API::DebuggableInfo>&&, const String& backendCommandsURL);
    void closeFromBackend();
    void show();
    void showConsole();
    void showResources();

    void sendMessageToFrontend(const String& message);

#if ENABLE(INSPECTOR_EXTENSIONS)
    WebInspectorUIExtensionControllerProxy* extensionController() const { return m_extensionController.get(); }
#endif
    
#if PLATFORM(MAC)
    NSWindow *window() const { return m_window.get(); }
    WKWebView *webView() const;

    const WebCore::FloatRect& sheetRect() const { return m_sheetRect; }

    void didBecomeActive();
#endif

#if PLATFORM(GTK)
    void updateWindowTitle(const CString&);
#endif

#if PLATFORM(WIN)
    LRESULT sizeChange();
    LRESULT onClose();

    static LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);
#endif

    void closeFromCrash();

private:
    RemoteWebInspectorUIProxy();
    RefPtr<WebPageProxy> protectedInspectorPage();

#if ENABLE(INSPECTOR_EXTENSIONS)
    RefPtr<WebInspectorUIExtensionControllerProxy> protectedExtensionController();
#endif

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

    // RemoteWebInspectorUIProxy messages.
    void frontendLoaded();
    void frontendDidClose();
    void reopen();
    void resetState();
    void bringToFront();
    void save(Vector<WebCore::InspectorFrontendClient::SaveData>&&, bool forceSaveAs);
    void load(const String& path, CompletionHandler<void(const String&)>&&);
    void pickColorFromScreen(CompletionHandler<void(const std::optional<WebCore::Color>&)>&&);
    void setSheetRect(const WebCore::FloatRect&);
    void setForcedAppearance(WebCore::InspectorFrontendClient::Appearance);
    void startWindowDrag();
    void openURLExternally(const String& url);
    void revealFileExternally(const String& path);
    void showCertificate(const WebCore::CertificateInfo&);
    void setInspectorPageDeveloperExtrasEnabled(bool);
    void sendMessageToBackend(const String& message);

    void createFrontendPageAndWindow();
    void closeFrontendPageAndWindow();

    // Platform implementations.
    WebPageProxy* platformCreateFrontendPageAndWindow();
    void platformCloseFrontendPageAndWindow();
    void platformResetState();
    void platformBringToFront();
    void platformSave(Vector<WebCore::InspectorFrontendClient::SaveData>&&, bool forceSaveAs);
    void platformLoad(const String& path, CompletionHandler<void(const String&)>&&);
    void platformPickColorFromScreen(CompletionHandler<void(const std::optional<WebCore::Color>&)>&&);
    void platformSetSheetRect(const WebCore::FloatRect&);
    void platformSetForcedAppearance(WebCore::InspectorFrontendClient::Appearance);
    void platformStartWindowDrag();
    void platformOpenURLExternally(const String& url);
    void platformRevealFileExternally(const String& path);
    void platformShowCertificate(const WebCore::CertificateInfo&);

    WeakPtr<RemoteWebInspectorUIProxyClient> m_client;
    WeakPtr<WebPageProxy> m_inspectorPage;

#if ENABLE(INSPECTOR_EXTENSIONS)
    RefPtr<WebInspectorUIExtensionControllerProxy> m_extensionController;
#endif
    
    Ref<API::DebuggableInfo> m_debuggableInfo;
    String m_backendCommandsURL;

#if PLATFORM(MAC)
    RetainPtr<WKInspectorViewController> m_inspectorView;
    RetainPtr<NSWindow> m_window;
    RetainPtr<WKRemoteWebInspectorUIProxyObjCAdapter> m_objCAdapter;
    HashMap<String, RetainPtr<NSURL>> m_suggestedToActualURLMap;
    WebCore::FloatRect m_sheetRect;
#endif
#if PLATFORM(GTK)
    GWeakPtr<GtkWidget> m_webView;
    GWeakPtr<GtkWidget> m_window;
#endif
#if PLATFORM(WIN)
    HWND m_frontendHandle;
    RefPtr<WebView> m_webView;
#endif
};

} // namespace WebKit
