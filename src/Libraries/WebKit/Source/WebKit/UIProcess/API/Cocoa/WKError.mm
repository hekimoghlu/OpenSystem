/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 31, 2024.
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
#import "config.h"
#import "WKErrorInternal.h"

#import <WebCore/LocalizedStrings.h>
#import <wtf/RetainPtr.h>
#import <wtf/text/WTFString.h>

NSString * const WKErrorDomain = @"WKErrorDomain";
NSString * const _WKLegacyErrorDomain = @"WebKitErrorDomain";

NSString * const _WKJavaScriptExceptionMessageErrorKey = @"WKJavaScriptExceptionMessage";
NSString * const _WKJavaScriptExceptionLineNumberErrorKey = @"WKJavaScriptExceptionLineNumber";
NSString * const _WKJavaScriptExceptionColumnNumberErrorKey = @"WKJavaScriptExceptionColumnNumber";
NSString * const _WKJavaScriptExceptionSourceURLErrorKey = @"WKJavaScriptExceptionSourceURL";

NSString *localizedDescriptionForErrorCode(WKErrorCode errorCode)
{
    switch (errorCode) {
    case WKErrorUnknown:
        return WEB_UI_STRING("An unknown error occurred", "WKErrorUnknown description");

    case WKErrorWebContentProcessTerminated:
        return WEB_UI_STRING("The Web Content process was terminated", "WKErrorWebContentProcessTerminated description");

    case WKErrorWebViewInvalidated:
        return WEB_UI_STRING("The WKWebView was invalidated", "WKErrorWebViewInvalidated description");

    case WKErrorJavaScriptExceptionOccurred:
        return WEB_UI_STRING("A JavaScript exception occurred", "WKErrorJavaScriptExceptionOccurred description");

    case WKErrorJavaScriptResultTypeIsUnsupported:
        return WEB_UI_STRING("JavaScript execution returned a result of an unsupported type", "WKErrorJavaScriptResultTypeIsUnsupported description");

    case WKErrorContentRuleListStoreLookUpFailed:
        return WEB_UI_STRING("Looking up a WKContentRuleList failed", "WKErrorContentRuleListStoreLookupFailed description");

    case WKErrorContentRuleListStoreVersionMismatch:
        return WEB_UI_STRING("Looking up a WKContentRuleList found a binary that is incompatible", "WKErrorContentRuleListStoreVersionMismatch description");

    case WKErrorContentRuleListStoreCompileFailed:
        return WEB_UI_STRING("Compiling a WKContentRuleList failed", "WKErrorContentRuleListStoreCompileFailed description");

    case WKErrorContentRuleListStoreRemoveFailed:
        return WEB_UI_STRING("Removing a WKContentRuleList failed", "WKErrorContentRuleListStoreRemoveFailed description");

    case WKErrorAttributedStringContentFailedToLoad:
        return WEB_UI_STRING("Attributed string content failed to load", "WKErrorAttributedStringContentFailedToLoad description");

    case WKErrorAttributedStringContentLoadTimedOut:
        return WEB_UI_STRING("Timed out while loading attributed string content", "WKErrorAttributedStringContentLoadTimedOut description");

    case WKErrorJavaScriptInvalidFrameTarget:
        return WEB_UI_STRING("JavaScript execution targeted an invalid frame", "WKErrorJavaScriptInvalidFrameTarget description");

    case WKErrorNavigationAppBoundDomain:
        return WEB_UI_STRING("Attempted to navigate away from an app-bound domain or navigate after using restricted APIs", "WKErrorNavigationAppBoundDomain description");

    case WKErrorJavaScriptAppBoundDomain:
        return WEB_UI_STRING("JavaScript execution targeted a frame that is not in an app-bound domain", "WKErrorJavaScriptAppBoundDomain description");

    case WKErrorDuplicateCredential:
        return WEB_UI_STRING("This credential is already present", "WKErrorDuplicateCredential description");

    case WKErrorMalformedCredential:
        return WEB_UI_STRING("This credential is malformed", "WKErrorMalformedCredential description");
            
    case WKErrorCredentialNotFound:
        return WEB_UI_STRING("Credential could not be found", "WKErrorCredentialNotFound description");
    }
}

RetainPtr<NSError> createNSError(WKErrorCode errorCode, NSError* underlyingError)
{
    NSDictionary *userInfo = nil;
    if (underlyingError)
        userInfo = @{ NSLocalizedDescriptionKey: localizedDescriptionForErrorCode(errorCode), NSUnderlyingErrorKey: underlyingError };
    else
        userInfo = @{ NSLocalizedDescriptionKey: localizedDescriptionForErrorCode(errorCode) };

    return adoptNS([[NSError alloc] initWithDomain:WKErrorDomain code:errorCode userInfo:userInfo]);
}
