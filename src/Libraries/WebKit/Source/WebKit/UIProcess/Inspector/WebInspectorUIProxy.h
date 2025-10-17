/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 17, 2024.
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
#include "Connection.h"
#include "DebuggableInfoData.h"
#include "MessageReceiver.h"
#include "WebInspectorUtilities.h"
#include "WebPageProxy.h"
#include "WebPageProxyIdentifier.h"
#include <JavaScriptCore/InspectorFrontendChannel.h>
#include <WebCore/Color.h>
#include <WebCore/FloatRect.h>
#include <WebCore/InspectorClient.h>
#include <WebCore/InspectorFrontendClient.h>
#include <wtf/CheckedPtr.h>
#include <wtf/Forward.h>
#include <wtf/RefPtr.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

#if PLATFORM(MAC)
#include "WKGeometry.h"
#include <WebCore/IntRect.h>
#include <wtf/HashMap.h>
#include <wtf/RetainPtr.h>
#include <wtf/RunLoop.h>

OBJC_CLASS NSString;
OBJC_CLASS NSURL;
OBJC_CLASS NSView;
OBJC_CLASS NSWindow;
OBJC_CLASS WKWebInspectorUIProxyObjCAdapter;
OBJC_CLASS WKInspectorViewController;
#elif PLATFORM(WIN)
#include "WebView.h"
#elif PLATFORM(GTK)
#include <wtf/glib/GWeakPtr.h>
#elif PLATFORM(WPE)
#include "WPEWebView.h"
#include <wtf/glib/GRefPtr.h>
typedef struct _WPEToplevel WPEToplevel;
#endif

namespace WebCore {
class CertificateInfo;
}

namespace API {
class InspectorClient;
}

namespace WebKit {

class WebFrameProxy;
class WebInspectorUIProxyClient;
class WebPreferences;
#if ENABLE(INSPECTOR_EXTENSIONS)
class WebInspectorUIExtensionControllerProxy;
#endif

enum class AttachmentSide {
    Bottom,
    Right,
    Left,
};

class WebInspectorUIProxy
    : public API::ObjectImpl<API::Object::Type::Inspector>
    , public IPC::MessageReceiver
    , public Inspector::FrontendChannel
#if PLATFORM(WIN)
    , public WebCore::WindowMessageListener
#endif
{
public:
    static Ref<WebInspectorUIProxy> create(WebPageProxy& inspectedPage)
    {
        return adoptRef(*new WebInspectorUIProxy(inspectedPage));
    }

    explicit WebInspectorUIProxy(WebPageProxy&);
    virtual ~WebInspectorUIProxy();

    void ref() const final { API::ObjectImpl<API::Object::Type::Inspector>::ref(); }
    void deref() const final { API::ObjectImpl<API::Object::Type::Inspector>::deref(); }

    void invalidate();

    API::InspectorClient& inspectorClient() { return *m_inspectorClient; }
    void setInspectorClient(std::unique_ptr<API::InspectorClient>&&);

    // Public APIs
    RefPtr<WebPageProxy> protectedInspectedPage() const { return m_inspectedPage.get(); }
    RefPtr<WebPageProxy> protectedInspectorPage() const { return m_inspectorPage.get(); }

#if ENABLE(INSPECTOR_EXTENSIONS)
    WebInspectorUIExtensionControllerProxy* extensionController() const { return m_extensionController.get(); }
    RefPtr<WebInspectorUIExtensionControllerProxy> protectedExtensionController() const;
#endif

    bool isConnected() const { return !!m_inspectorPage; }
    bool isVisible() const { return m_isVisible; }
    bool isFront();

    void connect();

    void show();
    void hide();
    void close();
    void closeForCrash();
    void reopen();
    void resetState();

    void reset();
    void updateForNewPageProcess(WebPageProxy&);

#if PLATFORM(MAC)
    enum class InspectionTargetType { Local, Remote };
    static RetainPtr<NSWindow> createFrontendWindow(NSRect savedWindowFrame, InspectionTargetType, WebPageProxy* inspectedPage = nullptr);
    static void showSavePanel(NSWindow *, NSURL *, Vector<WebCore::InspectorFrontendClient::SaveData>&&, bool forceSaveAs, CompletionHandler<void(NSURL *)>&&);

    void didBecomeActive();

    void updateInspectorWindowTitle() const;
    void inspectedViewFrameDidChange(CGFloat = 0);
    void windowFrameDidChange();
    void windowFullScreenDidChange();

    void closeFrontendPage();
    void closeFrontendAfterInactivityTimerFired();

    void attachmentViewDidChange(NSView *oldView, NSView *newView);
    void attachmentWillMoveFromWindow(NSWindow *oldWindow);
    void attachmentDidMoveToWindow(NSWindow *newWindow);

    const WebCore::FloatRect& sheetRect() const { return m_sheetRect; }
#endif

#if PLATFORM(WIN)
    static void showSavePanelForSingleFile(HWND, Vector<WebCore::InspectorFrontendClient::SaveData>&&);
#endif

#if PLATFORM(GTK)
    GtkWidget* inspectorView() const { return m_inspectorView.get(); };
    void setClient(std::unique_ptr<WebInspectorUIProxyClient>&&);
#endif

    void showConsole();
    void showResources();
    void showMainResourceForFrame(WebCore::FrameIdentifier);
    void openURLExternally(const String& url);
    void revealFileExternally(const String& path);

    AttachmentSide attachmentSide() const { return m_attachmentSide; }
    bool isAttached() const { return m_isAttached; }
    void attachRight();
    void attachLeft();
    void attachBottom();
    void attach(AttachmentSide = AttachmentSide::Bottom);
    void detach();

    void setAttachedWindowHeight(unsigned);
    void setAttachedWindowWidth(unsigned);

    void setSheetRect(const WebCore::FloatRect&);

    void startWindowDrag();

    bool isProfilingPage() const { return m_isProfilingPage; }
    void togglePageProfiling();

    bool isElementSelectionActive() const { return m_elementSelectionActive; }
    void toggleElementSelection();

    bool isUnderTest() const { return m_underTest; }
    void markAsUnderTest() { m_underTest = true; }

    void setDiagnosticLoggingAvailable(bool);

    // Provided by platform WebInspectorUIProxy implementations.
    static String inspectorPageURL();
    static String inspectorTestPageURL();
    static bool isMainOrTestInspectorPage(const URL&);
    static DebuggableInfoData infoForLocalDebuggable();

    static const unsigned minimumWindowWidth;
    static const unsigned minimumWindowHeight;

    static const unsigned initialWindowWidth;
    static const unsigned initialWindowHeight;

    // Testing methods.
    void evaluateInFrontendForTesting(const String&);

private:
    void createFrontendPage();
    void closeFrontendPageAndWindow();

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

    // Inspector::FrontendChannel
    void sendMessageToFrontend(const String& message) override;
    ConnectionType connectionType() const override { return ConnectionType::Local; }

    RefPtr<WebPageProxy> platformCreateFrontendPage();
    void platformCreateFrontendWindow();
    void platformCloseFrontendPageAndWindow();

    void platformDidCloseForCrash();
    void platformInvalidate();
    void platformResetState();
    void platformBringToFront();
    void platformBringInspectedPageToFront();
    void platformHide();
    bool platformIsFront();
    void platformAttachAvailabilityChanged(bool);
    void platformSetForcedAppearance(WebCore::InspectorFrontendClient::Appearance);
    void platformOpenURLExternally(const String&);
    void platformInspectedURLChanged(const String&);
    void platformShowCertificate(const WebCore::CertificateInfo&);
    void platformAttach();
    void platformDetach();
    void platformSetAttachedWindowHeight(unsigned);
    void platformSetAttachedWindowWidth(unsigned);
    void platformSetSheetRect(const WebCore::FloatRect&);
    void platformStartWindowDrag();
    void platformRevealFileExternally(const String&);
    void platformSave(Vector<WebCore::InspectorFrontendClient::SaveData>&&, bool forceSaveAs);
    void platformLoad(const String& path, CompletionHandler<void(const String&)>&&);
    void platformPickColorFromScreen(CompletionHandler<void(const std::optional<WebCore::Color>&)>&&);

#if PLATFORM(MAC)
    bool platformCanAttach(bool webProcessCanAttach);
#elif PLATFORM(WPE)
    bool platformCanAttach(bool) { return false; }
#else
    bool platformCanAttach(bool webProcessCanAttach) { return webProcessCanAttach; }
#endif

    // Called by WebInspectorUIProxy messages
    void requestOpenLocalInspectorFrontend();
    void setFrontendConnection(IPC::Connection::Handle&&);

    void openLocalInspectorFrontend();
    void sendMessageToBackend(const String&);
    void frontendLoaded();
    void didClose();
    void bringToFront();
    void bringInspectedPageToFront();
    void attachAvailabilityChanged(bool);
    void setForcedAppearance(WebCore::InspectorFrontendClient::Appearance);
    void effectiveAppearanceDidChange(WebCore::InspectorFrontendClient::Appearance);
    void inspectedURLChanged(const String&);
    void showCertificate(const WebCore::CertificateInfo&);
    void setInspectorPageDeveloperExtrasEnabled(bool);
    void elementSelectionChanged(bool);
    void timelineRecordingChanged(bool);

    void setDeveloperPreferenceOverride(WebCore::InspectorClient::DeveloperPreference, std::optional<bool>);
#if ENABLE(INSPECTOR_NETWORK_THROTTLING)
    void setEmulatedConditions(std::optional<int64_t>&& bytesPerSecondLimit);
#endif

    void save(Vector<WebCore::InspectorFrontendClient::SaveData>&&, bool forceSaveAs);
    void load(const String& path, CompletionHandler<void(const String&)>&&);
    void pickColorFromScreen(CompletionHandler<void(const std::optional<WebCore::Color>&)>&&);

    bool canAttach() const { return m_canAttach; }
    bool shouldOpenAttached();

    void open();

    unsigned inspectionLevel() const;

    WebPreferences& inspectorPagePreferences() const;
    Ref<WebPreferences> protectedInspectorPagePreferences() const;

#if PLATFORM(MAC)
    void applyForcedAppearance();
#endif

#if PLATFORM(GTK) || PLATFORM(WPE)
    void updateInspectorWindowTitle() const;
#endif

#if PLATFORM(WIN)
    static LRESULT CALLBACK wndProc(HWND, UINT, WPARAM, LPARAM);
    bool registerWindowClass();
    void windowReceivedMessage(HWND, UINT, WPARAM, LPARAM) override;
#endif

    WeakPtr<WebPageProxy> m_inspectedPage;
    WeakPtr<WebPageProxy> m_inspectorPage;
    std::unique_ptr<API::InspectorClient> m_inspectorClient;
    WebPageProxyIdentifier m_inspectedPageIdentifier;

#if ENABLE(INSPECTOR_EXTENSIONS)
    RefPtr<WebInspectorUIExtensionControllerProxy> m_extensionController;
#endif
    
    bool m_underTest { false };
    bool m_isVisible { false };
    bool m_isAttached { false };
    bool m_canAttach { false };
    bool m_isProfilingPage { false };
    bool m_showMessageSent { false };
    bool m_ignoreFirstBringToFront { false };
    bool m_elementSelectionActive { false };
    bool m_ignoreElementSelectionChange { false };
    bool m_isActiveFrontend { false };
    bool m_isOpening { false };
    bool m_closing { false };

    AttachmentSide m_attachmentSide {AttachmentSide::Bottom};

#if PLATFORM(MAC)
    RetainPtr<WKInspectorViewController> m_inspectorViewController;
    RetainPtr<NSWindow> m_inspectorWindow;
    RetainPtr<WKWebInspectorUIProxyObjCAdapter> m_objCAdapter;
    HashMap<String, RetainPtr<NSURL>> m_suggestedToActualURLMap;
    RunLoop::Timer m_closeFrontendAfterInactivityTimer;
    String m_urlString;
    WebCore::FloatRect m_sheetRect;
    WebCore::InspectorFrontendClient::Appearance m_frontendAppearance { WebCore::InspectorFrontendClient::Appearance::System };
    bool m_isObservingContentLayoutRect { false };
#elif PLATFORM(GTK)
    std::unique_ptr<WebInspectorUIProxyClient> m_client;
    GWeakPtr<GtkWidget> m_inspectorView;
    GWeakPtr<GtkWidget> m_inspectorWindow;
    GtkWidget* m_headerBar { nullptr };
    String m_inspectedURLString;
#elif PLATFORM(WPE)
    RefPtr<WKWPE::View> m_inspectorView;
    GRefPtr<WPEToplevel> m_inspectorWindow;
#elif PLATFORM(WIN)
    HWND m_inspectedViewWindow { nullptr };
    HWND m_inspectedViewParentWindow { nullptr };
    HWND m_inspectorViewWindow { nullptr };
    HWND m_inspectorDetachWindow { nullptr };
    RefPtr<WebView> m_inspectorView;
#endif
};

} // namespace WebKit
