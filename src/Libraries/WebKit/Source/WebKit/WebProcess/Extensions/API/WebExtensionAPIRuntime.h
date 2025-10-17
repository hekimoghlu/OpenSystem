/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 4, 2023.
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

#include "JSWebExtensionAPIRuntime.h"
#include "JSWebExtensionAPIWebPageRuntime.h"
#include "WebExtensionAPIEvent.h"
#include "WebExtensionAPIObject.h"
#include "WebPageProxyIdentifier.h"

OBJC_CLASS NSDictionary;
OBJC_CLASS NSString;
OBJC_CLASS NSURL;

namespace WebKit {

class WebExtensionAPIPort;
class WebExtensionAPIRuntime;

class WebExtensionAPIRuntimeBase : public JSWebExtensionWrappable {
public:
    JSValue *reportError(NSString *errorMessage, JSGlobalContextRef, Function<void()>&& = nullptr);
    JSValue *reportError(NSString *errorMessage, WebExtensionCallbackHandler&);

private:
    friend class WebExtensionAPIRuntime;

    bool m_lastErrorAccessed = false;

#if PLATFORM(COCOA)
    RetainPtr<JSValue> m_lastError;
#endif
};

class WebExtensionAPIRuntime : public WebExtensionAPIObject, public WebExtensionAPIRuntimeBase {
    WEB_EXTENSION_DECLARE_JS_WRAPPER_CLASS(WebExtensionAPIRuntime, runtime, runtime);

public:
    WebExtensionAPIRuntime& runtime() const final { return const_cast<WebExtensionAPIRuntime&>(*this); }

#if PLATFORM(COCOA)
    bool isPropertyAllowed(const ASCIILiteral& propertyName, WebPage*);

    NSURL *getURL(NSString *resourcePath, NSString **outExceptionString);
    NSDictionary *getManifest();
    void getPlatformInfo(Ref<WebExtensionCallbackHandler>&&);
    void getBackgroundPage(Ref<WebExtensionCallbackHandler>&&);
    double getFrameId(JSValue *);

    void setUninstallURL(URL, Ref<WebExtensionCallbackHandler>&&);

    void openOptionsPage(Ref<WebExtensionCallbackHandler>&&);
    void reload();

    NSString *runtimeIdentifier();

    JSValue *lastError();

    void sendMessage(WebPageProxyIdentifier, WebFrame&, NSString *extensionID, NSString *messageJSON, NSDictionary *options, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    RefPtr<WebExtensionAPIPort> connect(WebPageProxyIdentifier, WebFrame&, JSContextRef, NSString *extensionID, NSDictionary *options, NSString **outExceptionString);

    void sendNativeMessage(WebFrame&, NSString *applicationID, NSString *messageJSON, Ref<WebExtensionCallbackHandler>&&);
    RefPtr<WebExtensionAPIPort> connectNative(WebPageProxyIdentifier, JSContextRef, NSString *applicationID);

    WebExtensionAPIEvent& onConnect();
    WebExtensionAPIEvent& onInstalled();
    WebExtensionAPIEvent& onMessage();
    WebExtensionAPIEvent& onStartup();
    WebExtensionAPIEvent& onConnectExternal();
    WebExtensionAPIEvent& onMessageExternal();

private:
    friend class WebExtensionAPIWebPageRuntime;

    static bool parseConnectOptions(NSDictionary *, std::optional<String>& name, NSString *sourceKey, NSString **outExceptionString);

    RefPtr<WebExtensionAPIEvent> m_onConnect;
    RefPtr<WebExtensionAPIEvent> m_onInstalled;
    RefPtr<WebExtensionAPIEvent> m_onMessage;
    RefPtr<WebExtensionAPIEvent> m_onStartup;
    RefPtr<WebExtensionAPIEvent> m_onConnectExternal;
    RefPtr<WebExtensionAPIEvent> m_onMessageExternal;
#endif
};

class WebExtensionAPIWebPageRuntime : public WebExtensionAPIObject, public WebExtensionAPIRuntimeBase {
    WEB_EXTENSION_DECLARE_JS_WRAPPER_CLASS(WebExtensionAPIWebPageRuntime, webPageRuntime, webPageRuntime);

public:
    WebExtensionAPIWebPageRuntime& runtime() const final { return const_cast<WebExtensionAPIWebPageRuntime&>(*this); }

#if PLATFORM(COCOA)
    void sendMessage(WebPage&, WebFrame&, NSString *extensionID, NSString *messageJSON, NSDictionary *options, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    RefPtr<WebExtensionAPIPort> connect(WebPage&, WebFrame&, JSContextRef, NSString *extensionID, NSDictionary *options, NSString **outExceptionString);
#endif
};

NSDictionary *toWebAPI(const WebExtensionMessageSenderParameters&);

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
