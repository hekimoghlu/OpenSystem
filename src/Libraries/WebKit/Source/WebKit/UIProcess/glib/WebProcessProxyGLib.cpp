/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 19, 2025.
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
#include "WebProcessProxy.h"

#include "UserMessage.h"
#include "WebProcessPool.h"
#include "WebsiteDataStore.h"
#include <WebCore/NotImplemented.h>
#include <signal.h>
#include <sys/types.h>
#include <wtf/FileSystem.h>
#include <wtf/glib/Sandbox.h>

namespace WebKit {
using namespace WebCore;

void WebProcessProxy::platformGetLaunchOptions(ProcessLauncher::LaunchOptions& launchOptions)
{
    launchOptions.extraInitializationData.set("enable-sandbox"_s, m_processPool->sandboxEnabled() ? "true"_s : "false"_s);

#if USE(ATSPI)
    launchOptions.extraInitializationData.set("accessibilityBusAddress"_s, m_processPool->accessibilityBusAddress());
    launchOptions.extraInitializationData.set("accessibilityBusName"_s, m_processPool->generateNextAccessibilityBusName());
#endif

    if (m_processPool->sandboxEnabled()) {
        // Prewarmed processes don't have a WebsiteDataStore yet, so use the primary WebsiteDataStore from the WebProcessPool.
        // The process won't be used if current WebsiteDataStore is different than the WebProcessPool primary one.
        RefPtr dataStore = isPrewarmed() ? WebsiteDataStore::defaultDataStore().ptr() : websiteDataStore();

        ASSERT(dataStore);
        launchOptions.extraInitializationData.set("mediaKeysDirectory"_s, dataStore->resolvedDirectories().mediaKeysStorageDirectory);

        launchOptions.extraSandboxPaths = m_processPool->sandboxPaths();
#if USE(ATSPI)
        if (shouldUseBubblewrap())
            launchOptions.extraInitializationData.set("sandboxedAccessibilityBusAddress"_s, m_processPool->sandboxedAccessibilityBusAddress());
#endif
    }
}

void WebProcessProxy::sendMessageToWebContextWithReply(UserMessage&& message, CompletionHandler<void(UserMessage&&)>&& completionHandler)
{
    if (const auto& userMessageHandler = m_processPool->userMessageHandler())
        userMessageHandler(WTFMove(message), WTFMove(completionHandler));
}

void WebProcessProxy::sendMessageToWebContext(UserMessage&& message)
{
    sendMessageToWebContextWithReply(WTFMove(message), [](UserMessage&&) { });
}

void WebProcessProxy::platformSuspendProcess()
{
    // FIXME: https://webkit.org/b/280014
    notImplemented();
}

void WebProcessProxy::platformResumeProcess()
{
    // FIXME: https://webkit.org/b/280014
    notImplemented();
}

} // namespace WebKit
