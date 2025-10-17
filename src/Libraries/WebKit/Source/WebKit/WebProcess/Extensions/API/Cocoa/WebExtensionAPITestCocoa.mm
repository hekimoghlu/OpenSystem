/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 13, 2025.
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
#import "WebExtensionAPITest.h"

#import "CocoaHelpers.h"
#import "MessageSenderInlines.h"
#import "WebExtensionAPINamespace.h"
#import "WebExtensionAPIWebPageNamespace.h"
#import "WebExtensionControllerMessages.h"
#import "WebExtensionControllerProxy.h"
#import "WebExtensionEventListenerType.h"
#import "WebFrame.h"
#import "WebPage.h"
#import "WebProcess.h"
#import <JavaScriptCore/APICast.h>
#import <JavaScriptCore/ScriptCallStack.h>
#import <JavaScriptCore/ScriptCallStackFactory.h>
#import <WebCore/LocalFrame.h>

#if ENABLE(WK_WEB_EXTENSIONS)

namespace WebKit {

static std::pair<String, unsigned> scriptLocation(JSContextRef context)
{
    auto callStack = Inspector::createScriptCallStack(toJS(context));
    if (const Inspector::ScriptCallFrame* frame = callStack->firstNonNativeCallFrame()) {
        auto sourceURL = frame->sourceURL();
        if (sourceURL.isEmpty())
            sourceURL = "global code"_s;
        return { sourceURL, frame->lineNumber() };
    }

    return { "unknown"_s, 0 };
}

void WebExtensionAPITest::notifyFail(JSContextRef context, NSString *message)
{
    auto location = scriptLocation(context);

    RefPtr page = toWebPage(context);
    if (!page)
        return;

    RefPtr webExtensionControllerProxy = page->webExtensionControllerProxy();
    if (!webExtensionControllerProxy)
        return;

    WebProcess::singleton().send(Messages::WebExtensionController::TestFinished(false, message, location.first, location.second), webExtensionControllerProxy->identifier());
}

void WebExtensionAPITest::notifyPass(JSContextRef context, NSString *message)
{
    auto location = scriptLocation(context);

    RefPtr page = toWebPage(context);
    if (!page)
        return;

    RefPtr webExtensionControllerProxy = page->webExtensionControllerProxy();
    if (!webExtensionControllerProxy)
        return;

    WebProcess::singleton().send(Messages::WebExtensionController::TestFinished(true, message, location.first, location.second), webExtensionControllerProxy->identifier());
}

void WebExtensionAPITest::sendMessage(JSContextRef context, NSString *message, JSValue *argument)
{
    auto location = scriptLocation(context);

    RefPtr page = toWebPage(context);
    if (!page)
        return;

    RefPtr webExtensionControllerProxy = page->webExtensionControllerProxy();
    if (!webExtensionControllerProxy)
        return;

    WebProcess::singleton().send(Messages::WebExtensionController::TestSentMessage(message, argument._toSortedJSONString, location.first, location.second), webExtensionControllerProxy->identifier());
}

WebExtensionAPIEvent& WebExtensionAPITest::onMessage()
{
    if (!m_onMessage)
        m_onMessage = WebExtensionAPIEvent::create(*this, WebExtensionEventListenerType::TestOnMessage);

    return *m_onMessage;
}

JSValue *WebExtensionAPITest::runWithUserGesture(WebFrame& frame, JSValue *function)
{
    RefPtr coreFrame = frame.protectedCoreLocalFrame();
    WebCore::UserGestureIndicator gestureIndicator(WebCore::IsProcessingUserGesture::Yes, coreFrame ? coreFrame->document() : nullptr);

    return [function callWithArguments:@[ ]];
}

bool WebExtensionAPITest::isProcessingUserGesture()
{
    return WebCore::UserGestureIndicator::processingUserGesture();
}

inline NSString *debugString(JSValue *value)
{
    if (value._isRegularExpression || value._isFunction)
        return value.toString;
    return value._toSortedJSONString ?: @"undefined";
}

void WebExtensionAPITest::log(JSContextRef context, JSValue *value)
{
    auto location = scriptLocation(context);

    RefPtr page = toWebPage(context);
    if (!page)
        return;

    RefPtr webExtensionControllerProxy = page->webExtensionControllerProxy();
    if (!webExtensionControllerProxy)
        return;

    WebProcess::singleton().send(Messages::WebExtensionController::TestLogMessage(debugString(value), location.first, location.second), webExtensionControllerProxy->identifier());
}

void WebExtensionAPITest::fail(JSContextRef context, NSString *message)
{
    assertTrue(context, false, message);
}

void WebExtensionAPITest::succeed(JSContextRef context, NSString *message)
{
    assertTrue(context, true, message);
}

void WebExtensionAPITest::assertTrue(JSContextRef context, bool actualValue, NSString *message)
{
    auto location = scriptLocation(context);

    RefPtr page = toWebPage(context);
    if (!page)
        return;

    RefPtr webExtensionControllerProxy = page->webExtensionControllerProxy();
    if (!webExtensionControllerProxy)
        return;

    WebProcess::singleton().send(Messages::WebExtensionController::TestResult(actualValue, message, location.first, location.second), webExtensionControllerProxy->identifier());
}

void WebExtensionAPITest::assertFalse(JSContextRef context, bool actualValue, NSString *message)
{
    assertTrue(context, !actualValue, message);
}

void WebExtensionAPITest::assertDeepEq(JSContextRef context, JSValue *actualValue, JSValue *expectedValue, NSString *message)
{
    NSString *expectedJSONValue = debugString(expectedValue);
    NSString *actualJSONValue = debugString(actualValue);

    // FIXME: Comparing JSON is a quick attempt that works, but can still fail due to any non-JSON values.
    // See Firefox's implementation: https://searchfox.org/mozilla-central/source/toolkit/components/extensions/child/ext-test.js#78
    BOOL strictEqual = [expectedValue isEqualToObject:actualValue];
    BOOL deepEqual = strictEqual || [expectedJSONValue isEqualToString:actualJSONValue];

    auto location = scriptLocation(context);

    RefPtr page = toWebPage(context);
    if (!page)
        return;

    RefPtr webExtensionControllerProxy = page->webExtensionControllerProxy();
    if (!webExtensionControllerProxy)
        return;

    WebProcess::singleton().send(Messages::WebExtensionController::TestEqual(deepEqual, expectedJSONValue, actualJSONValue, message, location.first, location.second), webExtensionControllerProxy->identifier());
}

static NSString *combineMessages(NSString *messageOne, NSString *messageTwo)
{
    if (messageOne.length && messageTwo.length)
        return [[messageOne stringByAppendingString:@"\n"] stringByAppendingString:messageTwo];
    if (messageOne.length && !messageTwo.length)
        return messageOne;
    return messageTwo;
}

static void assertEquals(JSContextRef context, bool result, NSString *expectedString, NSString *actualString, NSString *message)
{
    auto location = scriptLocation(context);

    RefPtr page = toWebPage(context);
    if (!page)
        return;

    RefPtr webExtensionControllerProxy = page->webExtensionControllerProxy();
    if (!webExtensionControllerProxy)
        return;

    WebProcess::singleton().send(Messages::WebExtensionController::TestEqual(result, expectedString, actualString, message, location.first, location.second), webExtensionControllerProxy->identifier());
}

void WebExtensionAPITest::assertEq(JSContextRef context, JSValue *actualValue, JSValue *expectedValue, NSString *message)
{
    NSString *expectedJSONValue = debugString(expectedValue);
    NSString *actualJSONValue = debugString(actualValue);

    BOOL strictEqual = [expectedValue isEqualToObject:actualValue];
    if (!strictEqual && [expectedJSONValue isEqualToString:actualJSONValue])
        actualJSONValue = [actualJSONValue stringByAppendingString:@" (different)"];

    assertEquals(context, strictEqual, expectedJSONValue, actualJSONValue, message);
}

JSValue *WebExtensionAPITest::assertRejects(JSContextRef context, JSValue *promise, JSValue *expectedError, NSString *message)
{
    __block JSValue *resolveCallback;
    JSValue *resultPromise = [JSValue valueWithNewPromiseInContext:promise.context fromExecutor:^(JSValue *resolve, JSValue *reject) {
        resolveCallback = resolve;
    }];

    // Wrap in a native promise for consistency.
    promise = [JSValue valueWithNewPromiseResolvedWithResult:promise inContext:promise.context];

    [promise _awaitThenableResolutionWithCompletionHandler:^(JSValue *result, JSValue *error) {
        if (result || !error) {
            assertEquals(context, false, expectedError ? debugString(expectedError) : @"(any error)", result ? debugString(result) : @"(no error)", combineMessages(message, @"Promise did not reject with an error"));
            [resolveCallback callWithArguments:nil];
            return;
        }

        JSValue *errorMessageValue = error.isObject && [error hasProperty:@"message"] ? error[@"message"] : error;

        if (!expectedError) {
            assertEquals(context, true, @"(any error)", debugString(errorMessageValue), combineMessages(message, @"Promise rejected with an error"));
            [resolveCallback callWithArguments:nil];
            return;
        }

        if (expectedError._isRegularExpression) {
            JSValue *testResult = [expectedError invokeMethod:@"test" withArguments:@[ errorMessageValue ]];
            assertEquals(context, testResult.toBool, debugString(expectedError), debugString(errorMessageValue), combineMessages(message, @"Promise rejected with an error that didn't match the regular expression"));
            [resolveCallback callWithArguments:nil];
            return;
        }

        assertEquals(context, [expectedError isEqualWithTypeCoercionToObject:errorMessageValue], debugString(expectedError), debugString(errorMessageValue), combineMessages(message, @"Promise rejected with an error that didn't equal"));
        [resolveCallback callWithArguments:nil];
    }];

    return resultPromise;
}

JSValue *WebExtensionAPITest::assertResolves(JSContextRef context, JSValue *promise, NSString *message)
{
    __block JSValue *resolveCallback;
    JSValue *resultPromise = [JSValue valueWithNewPromiseInContext:promise.context fromExecutor:^(JSValue *resolve, JSValue *reject) {
        resolveCallback = resolve;
    }];

    // Wrap in a native promise for consistency.
    promise = [JSValue valueWithNewPromiseResolvedWithResult:promise inContext:promise.context];

    [promise _awaitThenableResolutionWithCompletionHandler:^(JSValue *result, JSValue *error) {
        if (!error) {
            succeed(context, @"Promise resolved without an error");
            [resolveCallback callWithArguments:@[ result ]];
            return;
        }

        JSValue *errorMessageValue = error.isObject && [error hasProperty:@"message"] ? error[@"message"] : error;
        notifyFail(context, combineMessages(message, [NSString stringWithFormat:@"Promise rejected with an error: %@", debugString(errorMessageValue)]));

        [resolveCallback callWithArguments:nil];
    }];

    return resultPromise;
}

void WebExtensionAPITest::assertThrows(JSContextRef context, JSValue *function, JSValue *expectedError, NSString *message)
{
    [function callWithArguments:@[ ]];

    JSValue *exceptionValue = function.context.exception;
    if (!exceptionValue) {
        assertEquals(context, false, expectedError ? debugString(expectedError) : @"(any exception)", @"(no exception)", combineMessages(message, @"Function did not throw an exception"));
        return;
    }

    JSValue *exceptionMessageValue = exceptionValue.isObject && [exceptionValue hasProperty:@"message"] ? exceptionValue[@"message"] : exceptionValue;

    // Clear the exception since it was caught.
    function.context.exception = nil;

    if (!expectedError) {
        assertEquals(context, true, @"(any exception)", debugString(exceptionMessageValue), combineMessages(message, @"Function threw an exception"));
        return;
    }

    if (expectedError._isRegularExpression) {
        JSValue *testResult = [expectedError invokeMethod:@"test" withArguments:@[ exceptionMessageValue ]];
        assertEquals(context, testResult.toBool, debugString(expectedError), debugString(exceptionMessageValue), combineMessages(message, @"Function threw an exception that didn't match the regular expression"));
        return;
    }

    assertEquals(context, [expectedError isEqualWithTypeCoercionToObject:exceptionMessageValue], debugString(expectedError), debugString(exceptionMessageValue), combineMessages(message, @"Function threw an exception that didn't equal"));
}

JSValue *WebExtensionAPITest::assertSafe(JSContextRef context, JSValue *function, NSString *message)
{
    JSValue *result = [function callWithArguments:@[ ]];

    JSValue *exceptionValue = function.context.exception;
    if (!exceptionValue) {
        succeed(context, @"Function did not throw an exception");
        return result;
    }

    // Clear the exception since it was caught.
    function.context.exception = nil;

    JSValue *exceptionMessageValue = exceptionValue.isObject && [exceptionValue hasProperty:@"message"] ? exceptionValue[@"message"] : exceptionValue;
    notifyFail(context, combineMessages(message, [NSString stringWithFormat:@"Function threw an exception: %@", debugString(exceptionMessageValue)]));

    return [JSValue valueWithUndefinedInContext:function.context];
}

JSValue *WebExtensionAPITest::assertSafeResolve(JSContextRef context, JSValue *function, NSString *message)
{
    JSValue *result = assertSafe(context, function, message);
    if (!result._isThenable)
        return result;

    return assertResolves(context, result, message);
}

void WebExtensionContextProxy::dispatchTestMessageEvent(const String& message, const String& argumentJSON, WebExtensionContentWorldType contentWorldType)
{
    id argument = parseJSON(argumentJSON, JSONOptions::FragmentsAllowed);

    if (contentWorldType == WebExtensionContentWorldType::WebPage) {
        enumerateFramesAndWebPageNamespaceObjects([&](auto&, auto& namespaceObject) {
            namespaceObject.test().onMessage().invokeListenersWithArgument(message, argument);
        });

        return;
    }

    enumerateFramesAndNamespaceObjects([&](auto&, auto& namespaceObject) {
        namespaceObject.test().onMessage().invokeListenersWithArgument(message, argument);
    }, toDOMWrapperWorld(contentWorldType));
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
