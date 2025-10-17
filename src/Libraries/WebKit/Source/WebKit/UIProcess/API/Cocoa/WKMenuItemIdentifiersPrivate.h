/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 19, 2024.
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
#import <WebKit/WKFoundation.h>

#import <Foundation/Foundation.h>

WK_EXTERN NSString * const _WKMenuItemIdentifierCopy WK_API_AVAILABLE(macos(10.12), ios(10.0));
WK_EXTERN NSString * const _WKMenuItemIdentifierCopyImage WK_API_AVAILABLE(macos(10.12), ios(10.0));
WK_EXTERN NSString * const _WKMenuItemIdentifierCopyLink WK_API_AVAILABLE(macos(10.12), ios(10.0));
WK_EXTERN NSString * const _WKMenuItemIdentifierCopyLinkWithHighlight WK_API_AVAILABLE(macos(15.2), ios(18.2), visionos(2.2));
WK_EXTERN NSString * const _WKMenuItemIdentifierCopyMediaLink WK_API_AVAILABLE(macos(10.14), ios(12.0));
WK_EXTERN NSString * const _WKMenuItemIdentifierDownloadImage WK_API_AVAILABLE(macos(10.12), ios(10.0));
WK_EXTERN NSString * const _WKMenuItemIdentifierDownloadLinkedFile WK_API_AVAILABLE(macos(10.12), ios(10.0));
WK_EXTERN NSString * const _WKMenuItemIdentifierDownloadMedia WK_API_AVAILABLE(macos(10.14), ios(12.0));
WK_EXTERN NSString * const _WKMenuItemIdentifierGoBack WK_API_AVAILABLE(macos(10.12), ios(10.0));
WK_EXTERN NSString * const _WKMenuItemIdentifierGoForward WK_API_AVAILABLE(macos(10.12), ios(10.0));
WK_EXTERN NSString * const _WKMenuItemIdentifierInspectElement WK_API_AVAILABLE(macos(10.12), ios(10.0));
WK_EXTERN NSString * const _WKMenuItemIdentifierLookUp WK_API_AVAILABLE(macos(10.12), ios(10.0));
WK_EXTERN NSString * const _WKMenuItemIdentifierOpenFrameInNewWindow WK_API_AVAILABLE(macos(10.12), ios(10.0));
WK_EXTERN NSString * const _WKMenuItemIdentifierOpenImageInNewWindow WK_API_AVAILABLE(macos(10.12), ios(10.0));
WK_EXTERN NSString * const _WKMenuItemIdentifierOpenLink WK_API_AVAILABLE(macos(10.12), ios(10.0));
WK_EXTERN NSString * const _WKMenuItemIdentifierOpenLinkInNewWindow WK_API_AVAILABLE(macos(10.12), ios(10.0));
WK_EXTERN NSString * const _WKMenuItemIdentifierOpenMediaInNewWindow WK_API_AVAILABLE(macos(10.14), ios(12.0));
WK_EXTERN NSString * const _WKMenuItemIdentifierPaste WK_API_AVAILABLE(macos(10.12), ios(10.0));
WK_EXTERN NSString * const _WKMenuItemIdentifierReload WK_API_AVAILABLE(macos(10.12), ios(10.0));
WK_EXTERN NSString * const _WKMenuItemIdentifierRevealImage WK_API_AVAILABLE(macos(12.0), ios(15.0));
WK_EXTERN NSString * const _WKMenuItemIdentifierSearchWeb WK_API_AVAILABLE(macos(10.12), ios(10.0));
WK_EXTERN NSString * const _WKMenuItemIdentifierShowHideMediaControls WK_API_AVAILABLE(macos(10.12), ios(10.0));
WK_EXTERN NSString * const _WKMenuItemIdentifierShowHideMediaStats WK_API_AVAILABLE(macos(13.3), ios(16.4));
WK_EXTERN NSString * const _WKMenuItemIdentifierToggleEnhancedFullScreen WK_API_AVAILABLE(macos(10.14), ios(12.0));
WK_EXTERN NSString * const _WKMenuItemIdentifierToggleVideoViewer WK_API_AVAILABLE(macos(15.0), ios(18.0));
WK_EXTERN NSString * const _WKMenuItemIdentifierToggleFullScreen WK_API_AVAILABLE(macos(10.12), ios(10.0));

WK_EXTERN NSString * const _WKMenuItemIdentifierShareMenu WK_API_AVAILABLE(macos(10.12), ios(10.0));
WK_EXTERN NSString * const _WKMenuItemIdentifierSpeechMenu WK_API_AVAILABLE(macos(10.12), ios(10.0));

WK_EXTERN NSString * const _WKMenuItemIdentifierAddHighlightToCurrentQuickNote WK_API_AVAILABLE(macos(12.0), ios(15.0));
WK_EXTERN NSString * const _WKMenuItemIdentifierAddHighlightToNewQuickNote WK_API_AVAILABLE(macos(12.0), ios(15.0));

WK_EXTERN NSString * const _WKMenuItemIdentifierTranslate WK_API_AVAILABLE(macos(12.0), ios(15.0));
WK_EXTERN NSString * const _WKMenuItemIdentifierWritingTools WK_API_AVAILABLE(macos(15.0), ios(18.0), visionos(2.0));
WK_EXTERN NSString * const _WKMenuItemIdentifierCopySubject WK_API_AVAILABLE(macos(13.0), ios(16.0));

WK_EXTERN NSString * const _WKMenuItemIdentifierSpellingMenu WK_API_AVAILABLE(macos(13.0), ios(16.0));
WK_EXTERN NSString * const _WKMenuItemIdentifierShowSpellingPanel WK_API_AVAILABLE(macos(13.0), ios(16.0));
WK_EXTERN NSString * const _WKMenuItemIdentifierCheckSpelling WK_API_AVAILABLE(macos(13.0), ios(16.0));
WK_EXTERN NSString * const _WKMenuItemIdentifierCheckSpellingWhileTyping WK_API_AVAILABLE(macos(13.0), ios(16.0));
WK_EXTERN NSString * const _WKMenuItemIdentifierCheckGrammarWithSpelling WK_API_AVAILABLE(macos(13.0), ios(16.0));
WK_EXTERN NSString * const _WKMenuItemIdentifierPlayAllAnimations WK_API_AVAILABLE(macos(13.3), ios(16.4));
WK_EXTERN NSString * const _WKMenuItemIdentifierPauseAllAnimations WK_API_AVAILABLE(macos(13.3), ios(16.4));
WK_EXTERN NSString * const _WKMenuItemIdentifierPlayAnimation  WK_API_AVAILABLE(macos(14.0), ios(17.0));
WK_EXTERN NSString * const _WKMenuItemIdentifierPauseAnimation  WK_API_AVAILABLE(macos(14.0), ios(17.0));
