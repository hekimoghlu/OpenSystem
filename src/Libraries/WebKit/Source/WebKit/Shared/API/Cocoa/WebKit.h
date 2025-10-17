/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 24, 2022.
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
#import <WebKit/NSAttributedString.h>
#import <WebKit/WKBackForwardList.h>
#import <WebKit/WKBackForwardListItem.h>
#import <WebKit/WKContentRuleList.h>
#import <WebKit/WKContentRuleListStore.h>
#import <WebKit/WKContentWorld.h>
#import <WebKit/WKContextMenuElementInfo.h>
#import <WebKit/WKDownload.h>
#import <WebKit/WKDownloadDelegate.h>
#import <WebKit/WKError.h>
#import <WebKit/WKFindConfiguration.h>
#import <WebKit/WKFindResult.h>
#import <WebKit/WKFoundation.h>
#import <WebKit/WKFrameInfo.h>
#import <WebKit/WKHTTPCookieStore.h>
#import <WebKit/WKNavigation.h>
#import <WebKit/WKNavigationAction.h>
#import <WebKit/WKNavigationDelegate.h>
#import <WebKit/WKNavigationResponse.h>
#import <WebKit/WKOpenPanelParameters.h>
#import <WebKit/WKPDFConfiguration.h>
#import <WebKit/WKPreferences.h>
#import <WebKit/WKPreviewActionItem.h>
#import <WebKit/WKPreviewActionItemIdentifiers.h>
#import <WebKit/WKPreviewElementInfo.h>
#import <WebKit/WKProcessPool.h>
#import <WebKit/WKScriptMessage.h>
#import <WebKit/WKScriptMessageHandler.h>
#import <WebKit/WKScriptMessageHandlerWithReply.h>
#import <WebKit/WKSecurityOrigin.h>
#import <WebKit/WKSnapshotConfiguration.h>
#import <WebKit/WKUIDelegate.h>
#import <WebKit/WKURLSchemeHandler.h>
#import <WebKit/WKURLSchemeTask.h>
#import <WebKit/WKUserContentController.h>
#import <WebKit/WKUserScript.h>
#import <WebKit/WKWebExtension.h>
#import <WebKit/WKWebExtensionAction.h>
#import <WebKit/WKWebExtensionCommand.h>
#import <WebKit/WKWebExtensionContext.h>
#import <WebKit/WKWebExtensionController.h>
#import <WebKit/WKWebExtensionControllerConfiguration.h>
#import <WebKit/WKWebExtensionControllerDelegate.h>
#import <WebKit/WKWebExtensionDataRecord.h>
#import <WebKit/WKWebExtensionDataType.h>
#import <WebKit/WKWebExtensionMatchPattern.h>
#import <WebKit/WKWebExtensionMessagePort.h>
#import <WebKit/WKWebExtensionPermission.h>
#import <WebKit/WKWebExtensionTab.h>
#import <WebKit/WKWebExtensionTabConfiguration.h>
#import <WebKit/WKWebExtensionWindow.h>
#import <WebKit/WKWebExtensionWindowConfiguration.h>
#import <WebKit/WKWebView.h>
#import <WebKit/WKWebViewConfiguration.h>
#import <WebKit/WKWebpagePreferences.h>
#import <WebKit/WKWebsiteDataRecord.h>
#import <WebKit/WKWebsiteDataStore.h>
#import <WebKit/WKWindowFeatures.h>

#if !defined(TARGET_OS_MACCATALYST) || !TARGET_OS_MACCATALYST
#import <WebKit/WebKitLegacy.h>
#endif
