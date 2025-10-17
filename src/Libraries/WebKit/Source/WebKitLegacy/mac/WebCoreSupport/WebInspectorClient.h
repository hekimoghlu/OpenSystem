/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 6, 2023.
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
#import <JavaScriptCore/InspectorFrontendChannel.h>
#import <WebCore/FloatRect.h>
#import <WebCore/InspectorClient.h>
#import <WebCore/InspectorDebuggableType.h>
#import <WebCore/InspectorFrontendClientLocal.h>
#import <wtf/Forward.h>
#import <wtf/HashMap.h>
#import <wtf/RetainPtr.h>
#import <wtf/TZoneMallocInlines.h>
#import <wtf/WeakObjCPtr.h>
#import <wtf/WeakPtr.h>
#import <wtf/text/StringHash.h>
#import <wtf/text/WTFString.h>

OBJC_CLASS NSURL;
OBJC_CLASS WebInspectorRemoteChannel;
OBJC_CLASS WebInspectorWindowController;
OBJC_CLASS WebNodeHighlighter;
OBJC_CLASS WebView;

namespace WebCore {
class CertificateInfo;
class LocalFrame;
class Page;
}

class WebInspectorFrontendClient;

class WebInspectorClient final : public WebCore::InspectorClient, public Inspector::FrontendChannel {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(WebInspectorClient);
public:
    explicit WebInspectorClient(WebView *inspectedWebView);
    virtual ~WebInspectorClient();

    void inspectedPageDestroyed() override;

    Inspector::FrontendChannel* openLocalFrontend(WebCore::InspectorController*) override;
    void bringFrontendToFront() override;
    void didResizeMainFrame(WebCore::LocalFrame*) override;

    void highlight() override;
    void hideHighlight() override;

#if ENABLE(REMOTE_INSPECTOR)
    bool allowRemoteInspectionToPageDirectly() const override { return true; }
#endif

#if PLATFORM(IOS_FAMILY)
    void showInspectorIndication() override;
    void hideInspectorIndication() override;

    bool overridesShowPaintRects() const override { return true; }
    void setShowPaintRects(bool) override;
    void showPaintRect(const WebCore::FloatRect&) override;
#endif

    void didSetSearchingForNode(bool) override;

    void sendMessageToFrontend(const String&) override;
    ConnectionType connectionType() const override { return ConnectionType::Local; }

    bool inspectorStartsAttached();
    void setInspectorStartsAttached(bool);
    void deleteInspectorStartsAttached();

    bool inspectorAttachDisabled();
    void setInspectorAttachDisabled(bool);
    void deleteInspectorAttachDisabled();

    void windowFullScreenDidChange();

    bool canAttach();

    void releaseFrontend();

private:
    std::unique_ptr<WebCore::InspectorFrontendClientLocal::Settings> createFrontendSettings();

    WeakObjCPtr<WebView> m_inspectedWebView;
    RetainPtr<WebNodeHighlighter> m_highlighter;
    WeakPtr<WebCore::Page> m_frontendPage;
    std::unique_ptr<WebInspectorFrontendClient> m_frontendClient;
};


class WebInspectorFrontendClient : public WebCore::InspectorFrontendClientLocal {
public:
    WebInspectorFrontendClient(WebView*, WebInspectorWindowController*, WebCore::InspectorController*, WebCore::Page*, std::unique_ptr<Settings>);

    void attachAvailabilityChanged(bool);
    bool canAttach();

    void frontendLoaded() override;

    void startWindowDrag() override;

    String localizedStringsURL() const override;
    Inspector::DebuggableType debuggableType() const final { return Inspector::DebuggableType::Page; };
    String targetPlatformName() const final { return "macOS"_s; };
    String targetBuildVersion() const final { return "Unknown"_s; };
    String targetProductVersion() const final { return "Unknown"_s; };
    bool targetIsSimulator() const final { return false; }


    void bringToFront() override;
    void closeWindow() override;
    void reopen() override;
    void resetState() override;

    void setForcedAppearance(InspectorFrontendClient::Appearance) override;

    bool supportsDockSide(DockSide) override;
    void attachWindow(DockSide) override;
    void detachWindow() override;

    void setAttachedWindowHeight(unsigned height) override;
    void setAttachedWindowWidth(unsigned height) override;

#if !PLATFORM(IOS_FAMILY)
    const WebCore::FloatRect& sheetRect() const { return m_sheetRect; }
#endif
    void setSheetRect(const WebCore::FloatRect&) override;

    void inspectedURLChanged(const String& newURL) override;
    void showCertificate(const WebCore::CertificateInfo&) override;

#if ENABLE(INSPECTOR_TELEMETRY)
    bool supportsDiagnosticLogging() override;
    void logDiagnosticEvent(const String& eventName, const WebCore::DiagnosticLoggingClient::ValueDictionary&) override;
#endif

private:
    void updateWindowTitle() const;

    bool canSave(WebCore::InspectorFrontendClient::SaveMode) override;
    void save(Vector<WebCore::InspectorFrontendClient::SaveData>&&, bool base64Encoded) override;

#if !PLATFORM(IOS_FAMILY)
    WeakObjCPtr<WebView> m_inspectedWebView;
    RetainPtr<WebInspectorWindowController> m_frontendWindowController;
    String m_inspectedURL;
    HashMap<String, RetainPtr<NSURL>> m_suggestedToActualURLMap;
    WebCore::FloatRect m_sheetRect;
#endif
};
