/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 3, 2023.
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
#import <WebKit/WKDownloadDelegatePrivate.h>
#import <WebKit/WKHistoryDelegatePrivate.h>
#import <WebKit/WKNavigationPrivate.h>
#import <WebKit/WKProcessPoolPrivate.h>
#import <WebKit/WKUIDelegatePrivate.h>
#import <WebKit/WKWebExtensionActionPrivate.h>
#import <WebKit/WKWebExtensionCommandPrivate.h>
#import <WebKit/WKWebExtensionContextPrivate.h>
#import <WebKit/WKWebExtensionControllerConfigurationPrivate.h>
#import <WebKit/WKWebExtensionControllerDelegatePrivate.h>
#import <WebKit/WKWebExtensionControllerPrivate.h>
#import <WebKit/WKWebExtensionDataRecordPrivate.h>
#import <WebKit/WKWebExtensionMatchPatternPrivate.h>
#import <WebKit/WKWebExtensionMessagePortPrivate.h>
#import <WebKit/WKWebExtensionPermissionPrivate.h>
#import <WebKit/WKWebExtensionPrivate.h>
#import <WebKit/WKWebViewConfigurationPrivate.h>
#import <WebKit/WKWebViewPrivate.h>
#import <WebKit/_WKActivatedElementInfo.h>
#import <WebKit/_WKAttachment.h>
#import <WebKit/_WKContentWorldConfiguration.h>
#import <WebKit/_WKElementAction.h>
#import <WebKit/_WKFocusedElementInfo.h>
#import <WebKit/_WKFormInputSession.h>
#import <WebKit/_WKInputDelegate.h>
#import <WebKit/_WKPageLoadTiming.h>
#import <WebKit/_WKProcessPoolConfiguration.h>
#import <WebKit/_WKTargetedElementInfo.h>
#import <WebKit/_WKTargetedElementRequest.h>
#import <WebKit/_WKTextRun.h>
#import <WebKit/_WKThumbnailView.h>
#import <WebKit/_WKVisitedLinkStore.h>
#import <WebKit/_WKWebPushAction.h>
#import <WebKit/_WKWebPushDaemonConnection.h>
#import <WebKit/_WKWebPushMessage.h>
#import <WebKit/_WKWebPushSubscriptionData.h>
