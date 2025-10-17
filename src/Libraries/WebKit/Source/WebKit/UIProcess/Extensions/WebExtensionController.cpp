/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 26, 2023.
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
#include "config.h"
#include "WebExtensionController.h"

#if ENABLE(WK_WEB_EXTENSIONS)

#include "WebExtensionControllerParameters.h"
#include "WebExtensionControllerProxyMessages.h"
#include "WebPageProxy.h"
#if PLATFORM(COCOA)
#include <wtf/BlockPtr.h>
#endif
#include <wtf/HashMap.h>
#include <wtf/NeverDestroyed.h>

namespace WebKit {

constexpr auto freshlyCreatedTimeout = 5_s;

static HashMap<WebExtensionControllerIdentifier, WeakPtr<WebExtensionController>>& webExtensionControllers()
{
    static MainThreadNeverDestroyed<HashMap<WebExtensionControllerIdentifier, WeakPtr<WebExtensionController>>> controllers;
    return controllers;
}

RefPtr<WebExtensionController> WebExtensionController::get(WebExtensionControllerIdentifier identifier)
{
    return webExtensionControllers().get(identifier).get();
}

WebExtensionController::WebExtensionController(Ref<WebExtensionControllerConfiguration> configuration)
    : m_configuration(configuration)
{
    ASSERT(!get(identifier()));
    webExtensionControllers().add(identifier(), *this);

    initializePlatform();

    // A freshly created extension controller will be used to determine if the startup event
    // should be fired for any loaded extensions during a brief time window. Start a timer
    // when the first extension is about to be loaded.

#if PLATFORM(COCOA)
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(freshlyCreatedTimeout.seconds() * NSEC_PER_SEC)), dispatch_get_main_queue(), makeBlockPtr([this, weakThis = WeakPtr { *this }] {
        if (!weakThis)
            return;

        m_freshlyCreated = false;
    }).get());
#endif
}

WebExtensionController::~WebExtensionController()
{
    webExtensionControllers().remove(identifier());
    unloadAll();
}

WebExtensionControllerParameters WebExtensionController::parameters() const
{
    return {
        .identifier = identifier(),
        .testingMode = inTestingMode(),
        .contextParameters = WTF::map(extensionContexts(), [](auto& context) {
            return context->parameters();
        })
    };
}

WebExtensionController::WebProcessProxySet WebExtensionController::allProcesses() const
{
    WebProcessProxySet result;

    for (Ref page : m_pages) {
        page->forEachWebContentProcess([&](auto& webProcess, auto pageID) {
            result.addVoid(webProcess);
        });
    }

    return result;
}

} // namespace WebKit

#endif // ENABLE(WK_WEB_EXTENSIONS)
