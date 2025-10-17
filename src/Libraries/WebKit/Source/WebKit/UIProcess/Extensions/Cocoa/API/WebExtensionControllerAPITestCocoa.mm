/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 18, 2022.
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
#if !__has_feature(objc_arc)
#error This file requires ARC. Add the "-fobjc-arc" compiler flag for this file.
#endif

#import "config.h"
#import "WebExtensionController.h"

#if ENABLE(WK_WEB_EXTENSIONS)

#import "CocoaHelpers.h"
#import "Logging.h"
#import "WKWebExtensionControllerDelegatePrivate.h"
#import "WKWebExtensionControllerInternal.h"

namespace WebKit {

void WebExtensionController::testResult(bool result, String message, String sourceURL, unsigned lineNumber)
{
    auto delegate = this->delegate();
    if ([delegate respondsToSelector:@selector(_webExtensionController:recordTestAssertionResult:withMessage:andSourceURL:lineNumber:)]) {
        [delegate _webExtensionController:wrapper() recordTestAssertionResult:result withMessage:message andSourceURL:sourceURL lineNumber:lineNumber];
        return;
    }

    if (message.isEmpty())
        message = "(no message)"_s;

    if (result) {
        RELEASE_LOG_INFO(Extensions, "Test assertion passed: %{public}@ (%{public}@:%{public}u)", (NSString *)message, (NSString *)sourceURL, lineNumber);
        return;
    }

    RELEASE_LOG_ERROR(Extensions, "Test assertion failed: %{public}@ (%{public}@:%{public}u)", (NSString *)message, (NSString *)sourceURL, lineNumber);
}

void WebExtensionController::testEqual(bool result, String expectedValue, String actualValue, String message, String sourceURL, unsigned lineNumber)
{
    auto delegate = this->delegate();
    if ([delegate respondsToSelector:@selector(_webExtensionController:recordTestEqualityResult:expectedValue:actualValue:withMessage:andSourceURL:lineNumber:)]) {
        [delegate _webExtensionController:wrapper() recordTestEqualityResult:result expectedValue:expectedValue actualValue:actualValue withMessage:message andSourceURL:sourceURL lineNumber:lineNumber];
        return;
    }

    if (message.isEmpty())
        message = "Expected equality of these values"_s;

    if (result) {
        RELEASE_LOG_INFO(Extensions, "Test equality passed: %{public}@: %{public}@ === %{public}@ (%{public}@:%{public}u)", (NSString *)message, (NSString *)expectedValue, (NSString *)actualValue, (NSString *)sourceURL, lineNumber);
        return;
    }

    RELEASE_LOG_ERROR(Extensions, "Test equality failed: %{public}@: %{public}@ !== %{public}@ (%{public}@:%{public}u)", (NSString *)message, (NSString *)expectedValue, (NSString *)actualValue, (NSString *)sourceURL, lineNumber);
}

void WebExtensionController::testLogMessage(String message, String sourceURL, unsigned lineNumber)
{
    auto delegate = this->delegate();
    if ([delegate respondsToSelector:@selector(_webExtensionController:logTestMessage:andSourceURL:lineNumber:)]) {
        [delegate _webExtensionController:wrapper() logTestMessage:message andSourceURL:sourceURL lineNumber:lineNumber];
        return;
    }

    if (message.isEmpty())
        message = "(no message)"_s;

    RELEASE_LOG_INFO(Extensions, "Test log: %{public}@ (%{public}@:%{public}u)", (NSString *)message, (NSString *)sourceURL, lineNumber);
}

void WebExtensionController::testSentMessage(String message, String argument, String sourceURL, unsigned lineNumber)
{
    auto delegate = this->delegate();
    if ([delegate respondsToSelector:@selector(_webExtensionController:receivedTestMessage:withArgument:andSourceURL:lineNumber:)]) {
        [delegate _webExtensionController:wrapper() receivedTestMessage:message withArgument:parseJSON(argument, JSONOptions::FragmentsAllowed) andSourceURL:sourceURL lineNumber:lineNumber];
        return;
    }

    RELEASE_LOG_INFO(Extensions, "Test sent message: %{public}@ %{public}@ (%{public}@:%{public}u)", (NSString *)message, (NSString *)argument, (NSString *)sourceURL, lineNumber);
}

void WebExtensionController::testFinished(bool result, String message, String sourceURL, unsigned lineNumber)
{
    auto delegate = this->delegate();
    if ([delegate respondsToSelector:@selector(_webExtensionController:recordTestFinishedWithResult:message:andSourceURL:lineNumber:)]) {
        [delegate _webExtensionController:wrapper() recordTestFinishedWithResult:result message:message andSourceURL:sourceURL lineNumber:lineNumber];
        return;
    }

    if (message.isEmpty())
        message = "(no message)"_s;

    if (result) {
        RELEASE_LOG_INFO(Extensions, "Test passed: %{public}@ (%{public}@:%{public}u)", (NSString *)message, (NSString *)sourceURL, lineNumber);
        return;
    }

    RELEASE_LOG_ERROR(Extensions, "Test failed: %{public}@ (%{public}@:%{public}u)", (NSString *)message, (NSString *)sourceURL, lineNumber);
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
