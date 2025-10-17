/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 3, 2023.
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

#include "JSWebExtensionAPIWebNavigation.h"
#include "WebExtensionAPIObject.h"
#include "WebExtensionAPIWebNavigationEvent.h"

namespace WebKit {

class WebPage;

class WebExtensionAPIWebNavigation : public WebExtensionAPIObject, public JSWebExtensionWrappable {
    WEB_EXTENSION_DECLARE_JS_WRAPPER_CLASS(WebExtensionAPIWebNavigation, webNavigation, webNavigation);

public:
#if PLATFORM(COCOA)
    WebExtensionAPIWebNavigationEvent& onBeforeNavigate();
    WebExtensionAPIWebNavigationEvent& onCommitted();
    WebExtensionAPIWebNavigationEvent& onDOMContentLoaded();
    WebExtensionAPIWebNavigationEvent& onCompleted();
    WebExtensionAPIWebNavigationEvent& onErrorOccurred();

    void getAllFrames(NSDictionary *details, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);
    void getFrame(NSDictionary *details, Ref<WebExtensionCallbackHandler>&&, NSString **outExceptionString);

private:
    RefPtr<WebExtensionAPIWebNavigationEvent> m_onBeforeNavigateEvent;
    RefPtr<WebExtensionAPIWebNavigationEvent> m_onCommittedEvent;
    RefPtr<WebExtensionAPIWebNavigationEvent> m_onDOMContentLoadedEvent;
    RefPtr<WebExtensionAPIWebNavigationEvent> m_onCompletedEvent;
    RefPtr<WebExtensionAPIWebNavigationEvent> m_onErrorOccurredEvent;
#endif
};

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
