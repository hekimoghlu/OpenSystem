/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 16, 2022.
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
#import <WebKit/WKError.h>

WK_EXTERN NSString * const _WKLegacyErrorDomain WK_API_AVAILABLE(macos(10.11), ios(8.3));

typedef NS_ENUM(NSInteger, _WKLegacyErrorCode) {
    _WKErrorCodeCannotShowURL WK_API_AVAILABLE(macos(13.3), ios(16.4)) = 101,
    _WKErrorCodeFrameLoadInterruptedByPolicyChange WK_API_AVAILABLE(macos(10.11), ios(9.0)) = 102,
    _WKErrorCodeFrameLoadBlockedByContentBlocker WK_API_AVAILABLE(macos(13.3), ios(16.4)) = 104,
    _WKErrorCodeFrameLoadBlockedByRestrictions WK_API_AVAILABLE(macos(10.15), ios(13.0)) = 106,
    _WKErrorCodeHTTPSUpgradeRedirectLoop WK_API_AVAILABLE(macos(14.0), ios(17.0)) = 304,
    _WKErrorCodeHTTPNavigationWithHTTPSOnly WK_API_AVAILABLE(macos(14.0), ios(17.0)) = 305,
    _WKLegacyErrorPlugInWillHandleLoad = 204,
} WK_API_AVAILABLE(macos(10.11), ios(8.3));

/*! @constant _WKJavaScriptExceptionMessageErrorKey Key in userInfo representing
 the exception message (as an NSString) for WKErrorJavaScriptExceptionOccurred errors. */
WK_EXTERN NSString * const _WKJavaScriptExceptionMessageErrorKey WK_API_AVAILABLE(macos(10.12), ios(10.0));

/*! @constant _WKJavaScriptExceptionLineNumberErrorKey Key in userInfo representing
 the exception line number (as an NSNumber) for WKErrorJavaScriptExceptionOccurred errors. */
WK_EXTERN NSString * const _WKJavaScriptExceptionLineNumberErrorKey WK_API_AVAILABLE(macos(10.12), ios(10.0));

/*! @constant _WKJavaScriptExceptionColumnNumberErrorKey Key in userInfo representing
 the exception column number (as an NSNumber) for WKErrorJavaScriptExceptionOccurred errors. */
WK_EXTERN NSString * const _WKJavaScriptExceptionColumnNumberErrorKey WK_API_AVAILABLE(macos(10.12), ios(10.0));

/*! @constant _WKJavaScriptExceptionSourceURLErrorKey Key in userInfo representing
 the exception source URL (as an NSURL) for WKErrorJavaScriptExceptionOccurred errors. */
WK_EXTERN NSString * const _WKJavaScriptExceptionSourceURLErrorKey WK_API_AVAILABLE(macos(10.12), ios(10.0));
