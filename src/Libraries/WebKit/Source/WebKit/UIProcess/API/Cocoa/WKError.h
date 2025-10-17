/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 28, 2023.
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

NS_ASSUME_NONNULL_BEGIN

/*! @constant WKErrorDomain Indicates a WebKit error. */
WK_EXTERN NSString * const WKErrorDomain WK_API_AVAILABLE(macos(10.10), ios(8.0));

/*! @enum WKErrorCode
 @abstract Constants used by NSError to indicate errors in the WebKit domain.
 @constant WKErrorUnknown                              Indicates that an unknown error occurred.
 @constant WKErrorWebContentProcessTerminated          Indicates that the Web Content process was terminated.
 @constant WKErrorWebViewInvalidated                   Indicates that the WKWebView was invalidated.
 @constant WKErrorJavaScriptExceptionOccurred          Indicates that a JavaScript exception occurred.
 @constant WKErrorJavaScriptResultTypeIsUnsupported    Indicates that the result of JavaScript execution could not be returned.
 @constant WKErrorContentRuleListStoreCompileFailed    Indicates that compiling a WKUserContentRuleList failed.
 @constant WKErrorContentRuleListStoreLookUpFailed     Indicates that looking up a WKUserContentRuleList failed.
 @constant WKErrorContentRuleListStoreRemoveFailed     Indicates that removing a WKUserContentRuleList failed.
 @constant WKErrorContentRuleListStoreVersionMismatch  Indicates that the WKUserContentRuleList version did not match the latest.
 @constant WKErrorAttributedStringContentFailedToLoad  Indicates that the attributed string content failed to load.
 @constant WKErrorAttributedStringContentLoadTimedOut  Indicates that loading attributed string content timed out.
 @constant WKErrorNavigationAppBoundDomain  Indicates that a navigation failed due to an app-bound domain restriction.
 @constant WKErrorJavaScriptAppBoundDomain  Indicates that JavaScript execution failed due to an app-bound domain restriction.
 */
typedef NS_ENUM(NSInteger, WKErrorCode) {
    WKErrorUnknown = 1,
    WKErrorWebContentProcessTerminated,
    WKErrorWebViewInvalidated,
    WKErrorJavaScriptExceptionOccurred,
    WKErrorJavaScriptResultTypeIsUnsupported WK_API_AVAILABLE(macos(10.11), ios(9.0)),
    WKErrorContentRuleListStoreCompileFailed WK_API_AVAILABLE(macos(10.13), ios(11.0)),
    WKErrorContentRuleListStoreLookUpFailed WK_API_AVAILABLE(macos(10.13), ios(11.0)),
    WKErrorContentRuleListStoreRemoveFailed WK_API_AVAILABLE(macos(10.13), ios(11.0)),
    WKErrorContentRuleListStoreVersionMismatch WK_API_AVAILABLE(macos(10.13), ios(11.0)),
    WKErrorAttributedStringContentFailedToLoad WK_API_AVAILABLE(macos(10.15), ios(13.0)),
    WKErrorAttributedStringContentLoadTimedOut WK_API_AVAILABLE(macos(10.15), ios(13.0)),
    WKErrorJavaScriptInvalidFrameTarget WK_API_AVAILABLE(macos(11.0), ios(14.0)),
    WKErrorNavigationAppBoundDomain WK_API_AVAILABLE(macos(11.0), ios(14.0)),
    WKErrorJavaScriptAppBoundDomain WK_API_AVAILABLE(macos(11.0), ios(14.0)),
    WKErrorDuplicateCredential WK_API_AVAILABLE(macos(13.0), ios(16.0)),
    WKErrorMalformedCredential WK_API_AVAILABLE(macos(13.0), ios(16.0)),
    WKErrorCredentialNotFound WK_API_AVAILABLE(macos(13.0), ios(16.0)),
} WK_API_AVAILABLE(macos(10.10), ios(8.0));

NS_ASSUME_NONNULL_END
