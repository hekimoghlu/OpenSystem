/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 6, 2023.
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
#import "WebExtensionAPIDevToolsNetwork.h"

#if ENABLE(WK_WEB_EXTENSIONS) && ENABLE(INSPECTOR_EXTENSIONS)

#import "CocoaHelpers.h"
#import "JSWebExtensionWrapper.h"
#import "MessageSenderInlines.h"
#import "WebExtensionAPIEvent.h"
#import "WebExtensionAPINamespace.h"

namespace WebKit {

WebExtensionAPIEvent& WebExtensionAPIDevToolsNetwork::onNavigated()
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/devtools/network/onNavigated

    if (!m_onNavigated)
        m_onNavigated = WebExtensionAPIEvent::create(*this, WebExtensionEventListenerType::DevToolsNetworkOnNavigated);

    return *m_onNavigated;
}

void WebExtensionContextProxy::dispatchDevToolsNetworkNavigatedEvent(const URL& url)
{
    // Documentation: https://developer.mozilla.org/docs/Mozilla/Add-ons/WebExtensions/API/devtools/network/onNavigated

    NSString *urlString = url.string();

    enumerateNamespaceObjects([&](auto& namespaceObject) {
        namespaceObject.devtools().network().onNavigated().invokeListenersWithArgument(urlString);
    });
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS) && ENABLE(INSPECTOR_EXTENSIONS)
