/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 7, 2023.
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

#include "JSWebExtensionAPIWebRequest.h"
#include "WebExtensionAPIObject.h"
#include "WebExtensionAPIWebRequestEvent.h"

namespace WebKit {

class WebPage;

class WebExtensionAPIWebRequest : public WebExtensionAPIObject, public JSWebExtensionWrappable {
    WEB_EXTENSION_DECLARE_JS_WRAPPER_CLASS(WebExtensionAPIWebRequest, webRequest, webRequest);

public:
#if PLATFORM(COCOA)
    WebExtensionAPIWebRequestEvent& onBeforeRequest();
    WebExtensionAPIWebRequestEvent& onBeforeSendHeaders();
    WebExtensionAPIWebRequestEvent& onSendHeaders();
    WebExtensionAPIWebRequestEvent& onHeadersReceived();
    WebExtensionAPIWebRequestEvent& onAuthRequired();
    WebExtensionAPIWebRequestEvent& onBeforeRedirect();
    WebExtensionAPIWebRequestEvent& onResponseStarted();
    WebExtensionAPIWebRequestEvent& onCompleted();
    WebExtensionAPIWebRequestEvent& onErrorOccurred();

private:
    RefPtr<WebExtensionAPIWebRequestEvent> m_onBeforeRequestEvent;
    RefPtr<WebExtensionAPIWebRequestEvent> m_onBeforeSendHeadersEvent;
    RefPtr<WebExtensionAPIWebRequestEvent> m_onSendHeadersEvent;
    RefPtr<WebExtensionAPIWebRequestEvent> m_onHeadersReceivedEvent;
    RefPtr<WebExtensionAPIWebRequestEvent> m_onAuthRequiredEvent;
    RefPtr<WebExtensionAPIWebRequestEvent> m_onBeforeRedirectEvent;
    RefPtr<WebExtensionAPIWebRequestEvent> m_onResponseStartedEvent;
    RefPtr<WebExtensionAPIWebRequestEvent> m_onCompletedEvent;
    RefPtr<WebExtensionAPIWebRequestEvent> m_onErrorOccurredEvent;

#endif
};

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
