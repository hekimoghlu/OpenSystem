/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 21, 2023.
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

#if ENABLE(WK_WEB_EXTENSIONS)

#include "JSWebExtensionAPITest.h"
#include "WebExtensionAPIEvent.h"
#include "WebExtensionAPIObject.h"
#include "WebExtensionAPIWebNavigationEvent.h"

OBJC_CLASS NSString;

namespace WebKit {

class WebExtensionAPITest : public WebExtensionAPIObject, public JSWebExtensionWrappable {
    WEB_EXTENSION_DECLARE_JS_WRAPPER_CLASS(WebExtensionAPITest, test, test);

public:
#if PLATFORM(COCOA)
    void notifyFail(JSContextRef, NSString *message);
    void notifyPass(JSContextRef, NSString *message);

    void sendMessage(JSContextRef, NSString *message, JSValue *argument);
    WebExtensionAPIEvent& onMessage();

    JSValue *runWithUserGesture(WebFrame&, JSValue *function);
    bool isProcessingUserGesture();

    void log(JSContextRef, JSValue *);

    void fail(JSContextRef, NSString *message);
    void succeed(JSContextRef, NSString *message);

    void assertTrue(JSContextRef, bool testValue, NSString *message);
    void assertFalse(JSContextRef, bool testValue, NSString *message);

    void assertDeepEq(JSContextRef, JSValue *actualValue, JSValue *expectedValue, NSString *message);
    void assertEq(JSContextRef, JSValue *actualValue, JSValue *expectedValue, NSString *message);

    JSValue *assertRejects(JSContextRef, JSValue *promise, JSValue *expectedError, NSString *message);
    JSValue *assertResolves(JSContextRef, JSValue *promise, NSString *message);

    void assertThrows(JSContextRef, JSValue *function, JSValue *expectedError, NSString *message);
    JSValue *assertSafe(JSContextRef, JSValue *function, NSString *message);

    JSValue *assertSafeResolve(JSContextRef, JSValue *function, NSString *message);

private:
    RefPtr<WebExtensionAPIEvent> m_onMessage;
#endif
};

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
