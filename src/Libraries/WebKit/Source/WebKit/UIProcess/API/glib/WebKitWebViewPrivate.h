/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 3, 2022.
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

#include "EditingRange.h"
#include "RendererBufferFormat.h"
#include "UserMessage.h"
#include "WebContextMenuItemData.h"
#include "WebEvent.h"
#include "WebHitTestResultData.h"
#include "WebImage.h"
#include "WebKitWebView.h"
#include "WebPageProxy.h"
#include <WebCore/CompositionUnderline.h>
#include <WebCore/IntRect.h>
#include <WebCore/LinkIcon.h>
#include <WebCore/MediaProducer.h>
#include <WebCore/ResourceRequest.h>
#include <wtf/CompletionHandler.h>
#include <wtf/text/CString.h>

namespace WebKit {
class WebKitWebResourceLoadManager;
}

WebKit::WebPageProxy& webkitWebViewGetPage(WebKitWebView*);
void webkitWebViewWillStartLoad(WebKitWebView*);
void webkitWebViewLoadChanged(WebKitWebView*, WebKitLoadEvent);
void webkitWebViewLoadFailed(WebKitWebView*, WebKitLoadEvent, const char* failingURI, GError*);
void webkitWebViewLoadFailedWithTLSErrors(WebKitWebView*, const char* failingURI, GError*, GTlsCertificateFlags, GTlsCertificate*);
#if PLATFORM(GTK)
void webkitWebViewGetLoadDecisionForIcon(WebKitWebView*, const WebCore::LinkIcon&, Function<void(bool)>&&);
void webkitWebViewSetIcon(WebKitWebView*, const WebCore::LinkIcon&, API::Data&);
#endif
RefPtr<WebKit::WebPageProxy> webkitWebViewCreateNewPage(WebKitWebView*, Ref<API::PageConfiguration>&&, WebKitNavigationAction*);
void webkitWebViewReadyToShowPage(WebKitWebView*);
void webkitWebViewRunAsModal(WebKitWebView*);
void webkitWebViewClosePage(WebKitWebView*);
void webkitWebViewRunJavaScriptAlert(WebKitWebView*, const CString& message, Function<void()>&& completionHandler);
void webkitWebViewRunJavaScriptConfirm(WebKitWebView*, const CString& message, Function<void(bool)>&& completionHandler);
void webkitWebViewRunJavaScriptPrompt(WebKitWebView*, const CString& message, const CString& defaultText, Function<void(const String&)>&& completionHandler);
void webkitWebViewRunJavaScriptBeforeUnloadConfirm(WebKitWebView*, const CString& message, Function<void(bool)>&& completionHandler);
bool webkitWebViewIsShowingScriptDialog(WebKitWebView*);
bool webkitWebViewIsScriptDialogRunning(WebKitWebView*, WebKitScriptDialog*);
String webkitWebViewGetCurrentScriptDialogMessage(WebKitWebView*);
void webkitWebViewSetCurrentScriptDialogUserInput(WebKitWebView*, const String&);
void webkitWebViewAcceptCurrentScriptDialog(WebKitWebView*);
void webkitWebViewDismissCurrentScriptDialog(WebKitWebView*);
std::optional<WebKitScriptDialogType> webkitWebViewGetCurrentScriptDialogType(WebKitWebView*);
void webkitWebViewMakePermissionRequest(WebKitWebView*, WebKitPermissionRequest*);
void webkitWebViewMakePolicyDecision(WebKitWebView*, WebKitPolicyDecisionType, WebKitPolicyDecision*);
void webkitWebViewMouseTargetChanged(WebKitWebView*, const WebKit::WebHitTestResultData&, OptionSet<WebKit::WebEventModifier>);
void webkitWebViewPrintFrame(WebKitWebView*, WebKit::WebFrameProxy*);
WebKit::WebKitWebResourceLoadManager* webkitWebViewGetWebResourceLoadManager(WebKitWebView*);
void webkitWebViewResourceLoadStarted(WebKitWebView*, WebKitWebResource*, WebCore::ResourceRequest&&);
void webkitWebViewRunFileChooserRequest(WebKitWebView*, WebKitFileChooserRequest*);
#if PLATFORM(GTK)
void webKitWebViewDidReceiveSnapshot(WebKitWebView*, uint64_t callbackID, WebKit::WebImage*);
#endif
void webkitWebViewMaximizeWindow(WebKitWebView*, CompletionHandler<void()>&&);
void webkitWebViewMinimizeWindow(WebKitWebView*, CompletionHandler<void()>&&);
void webkitWebViewRestoreWindow(WebKitWebView*, CompletionHandler<void()>&&);
#if ENABLE(FULLSCREEN_API)
bool webkitWebViewEnterFullScreen(WebKitWebView*);
bool webkitWebViewExitFullScreen(WebKitWebView*);
#endif
void webkitWebViewPopulateContextMenu(WebKitWebView*, const Vector<WebKit::WebContextMenuItemData>& proposedMenu, const WebKit::WebHitTestResultData&, GVariant*);
void webkitWebViewSubmitFormRequest(WebKitWebView*, WebKitFormSubmissionRequest*);
void webkitWebViewHandleAuthenticationChallenge(WebKitWebView*, WebKit::AuthenticationChallengeProxy*);
void webkitWebViewInsecureContentDetected(WebKitWebView*, WebKitInsecureContentEvent);
bool webkitWebViewEmitShowNotification(WebKitWebView*, WebKitNotification*);
void webkitWebViewWebProcessTerminated(WebKitWebView*, WebKitWebProcessTerminationReason);
void webkitWebViewIsPlayingAudioChanged(WebKitWebView*);
void webkitWebViewMediaCaptureStateDidChange(WebKitWebView*, WebCore::MediaProducer::MediaStateFlags);
void webkitWebViewSelectionDidChange(WebKitWebView*);
WebKitWebsiteDataManager* webkitWebViewGetWebsiteDataManager(WebKitWebView*);
void webkitWebViewPermissionStateQuery(WebKitWebView*, WebKitPermissionStateQuery*);

#if PLATFORM(GTK)
bool webkitWebViewEmitRunColorChooser(WebKitWebView*, WebKitColorChooserRequest*);
#endif

bool webkitWebViewShowOptionMenu(WebKitWebView*, const WebCore::IntRect&, WebKitOptionMenu*);

gboolean webkitWebViewAuthenticate(WebKitWebView*, WebKitAuthenticationRequest*);
gboolean webkitWebViewScriptDialog(WebKitWebView*, WebKitScriptDialog*);
gboolean webkitWebViewRunFileChooser(WebKitWebView*, WebKitFileChooserRequest*);
void webkitWebViewDidChangePageID(WebKitWebView*);
void webkitWebViewDidReceiveUserMessage(WebKitWebView*, WebKit::UserMessage&&, CompletionHandler<void(WebKit::UserMessage&&)>&&);

#if ENABLE(POINTER_LOCK)
void webkitWebViewRequestPointerLock(WebKitWebView*);
void webkitWebViewDenyPointerLockRequest(WebKitWebView*);
void webkitWebViewDidLosePointerLock(WebKitWebView*);
#endif

void webkitWebViewSetComposition(WebKitWebView*, const String&, const Vector<WebCore::CompositionUnderline>&, WebKit::EditingRange&&);
void webkitWebViewConfirmComposition(WebKitWebView*, const String&);
void webkitWebViewCancelComposition(WebKitWebView*, const String&);
void webkitWebViewDeleteSurrounding(WebKitWebView*, int offset, unsigned characterCount);
void webkitWebViewSetIsWebProcessResponsive(WebKitWebView*, bool);

guint createShowOptionMenuSignal(WebKitWebViewClass*);
guint createContextMenuSignal(WebKitWebViewClass*);

#if PLATFORM(GTK) || (PLATFORM(WPE) && ENABLE(WPE_PLATFORM))
WebKit::RendererBufferFormat webkitWebViewGetRendererBufferFormat(WebKitWebView*);
#endif
