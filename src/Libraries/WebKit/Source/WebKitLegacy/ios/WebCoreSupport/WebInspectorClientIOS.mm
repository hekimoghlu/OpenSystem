/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 4, 2023.
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
#import "WebInspectorClient.h"

#if PLATFORM(IOS_FAMILY)

#import "WebFrameInternal.h"
#import "WebInspector.h"
#import "WebNodeHighlighter.h"
#import "WebViewInternal.h"
#import <WebCore/CertificateInfo.h>
#import <WebCore/FloatRect.h>
#import <WebCore/InspectorController.h>
#import <WebCore/NotImplemented.h>
#import <WebCore/Page.h>
#import <WebCore/Settings.h>
#import <WebCore/WebCoreThread.h>

using namespace WebCore;

WebInspectorClient::WebInspectorClient(WebView* inspectedWebView)
    : m_inspectedWebView(inspectedWebView)
    , m_highlighter(adoptNS([[WebNodeHighlighter alloc] initWithInspectedWebView:inspectedWebView]))
{
}

WebInspectorClient::~WebInspectorClient() = default;

void WebInspectorClient::inspectedPageDestroyed()
{
}

Inspector::FrontendChannel* WebInspectorClient::openLocalFrontend(InspectorController*)
{
    // iOS does not have a local inspector, this should not be reached.
    ASSERT_NOT_REACHED();
    return nullptr;
}

void WebInspectorClient::bringFrontendToFront()
{
    // iOS does not have a local inspector, nothing to do here.
}

void WebInspectorClient::didResizeMainFrame(LocalFrame*)
{
    // iOS does not have a local inspector, nothing to do here.
}

void WebInspectorClient::highlight()
{
    [m_highlighter.get() highlight];
}

void WebInspectorClient::hideHighlight()
{
    [m_highlighter.get() hideHighlight];
}

void WebInspectorClient::showInspectorIndication()
{
    [m_inspectedWebView.get() setShowingInspectorIndication:YES];
}

void WebInspectorClient::hideInspectorIndication()
{
    [m_inspectedWebView.get() setShowingInspectorIndication:NO];
}

void WebInspectorClient::setShowPaintRects(bool)
{
    // FIXME: implement.
}

void WebInspectorClient::showPaintRect(const FloatRect&)
{
    // FIXME: need to do CALayer-based highlighting of paint rects.
}

void WebInspectorClient::didSetSearchingForNode(bool enabled)
{
    NSString *notificationName = enabled ? WebInspectorDidStartSearchingForNode : WebInspectorDidStopSearchingForNode;
    RunLoop::main().dispatch([notificationName = retainPtr(notificationName), inspector = retainPtr([m_inspectedWebView.get() inspector])] {
        [[NSNotificationCenter defaultCenter] postNotificationName:notificationName.get() object:inspector.get()];
    });
}

#pragma mark -
#pragma mark WebInspectorFrontendClient Implementation

WebInspectorFrontendClient::WebInspectorFrontendClient(WebView* inspectedWebView, WebInspectorWindowController* frontendWindowController, InspectorController* inspectedPageController, Page* frontendPage, std::unique_ptr<Settings> settings)
    : InspectorFrontendClientLocal(inspectedPageController, frontendPage, WTFMove(settings))
{
    // iOS does not have a local inspector, this should not be reached.
    notImplemented();
}

void WebInspectorFrontendClient::attachAvailabilityChanged(bool) { }
void WebInspectorFrontendClient::frontendLoaded() { }
String WebInspectorFrontendClient::localizedStringsURL() const { return String(); }
void WebInspectorFrontendClient::bringToFront() { }
void WebInspectorFrontendClient::closeWindow() { }
void WebInspectorFrontendClient::reopen() { }
void WebInspectorFrontendClient::resetState() { }
void WebInspectorFrontendClient::setForcedAppearance(InspectorFrontendClient::Appearance) { }
bool WebInspectorFrontendClient::supportsDockSide(DockSide) { return false; }
void WebInspectorFrontendClient::attachWindow(DockSide) { }
void WebInspectorFrontendClient::detachWindow() { }
void WebInspectorFrontendClient::setAttachedWindowHeight(unsigned) { }
void WebInspectorFrontendClient::setAttachedWindowWidth(unsigned) { }
void WebInspectorFrontendClient::setSheetRect(const FloatRect&) { }
void WebInspectorFrontendClient::startWindowDrag() { }
void WebInspectorFrontendClient::inspectedURLChanged(const String&) { }
void WebInspectorFrontendClient::showCertificate(const CertificateInfo&) { }
bool WebInspectorFrontendClient::supportsDiagnosticLogging() { return false; }
void WebInspectorFrontendClient::logDiagnosticEvent(const String& eventName, const WebCore::DiagnosticLoggingClient::ValueDictionary&) { }
void WebInspectorFrontendClient::updateWindowTitle() const { }
bool WebInspectorFrontendClient::canSave(WebCore::InspectorFrontendClient::SaveMode) { return false; }
void WebInspectorFrontendClient::save(Vector<InspectorFrontendClient::SaveData>&&, bool /* forceSaveAs */) { }

#endif // PLATFORM(IOS_FAMILY)
