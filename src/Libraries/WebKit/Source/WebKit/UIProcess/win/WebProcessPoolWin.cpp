/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 7, 2025.
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
#include "WebProcessPool.h"

#include "WebProcessCreationParameters.h"
#include <WebCore/NotImplemented.h>

#if ENABLE(REMOTE_INSPECTOR)
#include "AutomationClientWin.h"
#include <JavaScriptCore/RemoteInspector.h>
#include <JavaScriptCore/RemoteInspectorServer.h>
#include <WebCore/WebCoreBundleWin.h>
#include <wtf/text/StringToIntegerConversion.h>
#endif

namespace WebKit {

#if ENABLE(REMOTE_INSPECTOR)
static void initializeRemoteInspectorServer(StringView address)
{
    if (Inspector::RemoteInspectorServer::singleton().isRunning())
        return;

    auto pos = address.find(':');
    if (pos == notFound)
        return;

    auto host = address.left(pos);
    auto port = parseInteger<uint16_t>(address.substring(pos + 1));
    if (!port)
        return;

    auto backendCommands = WebCore::webKitBundlePath({ "WebKit.Resources"_s, "WebInspectorUI"_s, "Protocol"_s, "InspectorBackendCommands.js"_s });
    Inspector::RemoteInspector::singleton().setBackendCommandsPath(backendCommands);
    Inspector::RemoteInspectorServer::singleton().start(host.utf8().data(), port.value());
}
#endif

void WebProcessPool::platformInitialize(NeedsGlobalStaticInitialization)
{
#if ENABLE(REMOTE_INSPECTOR)
    if (const char* address = getenv("WEBKIT_INSPECTOR_SERVER"))
        initializeRemoteInspectorServer(StringView::fromLatin1(address));

    // Currently the socket port Remote Inspector can have only one client at most.
    // Therefore, if multiple process pools are created, the first one is targeted and the second and subsequent ones are ignored.
    if (!Inspector::RemoteInspector::singleton().client())
        setAutomationClient(WTF::makeUnique<AutomationClient>(*this));
#endif
}

void WebProcessPool::platformInitializeNetworkProcess(NetworkProcessCreationParameters&)
{
    notImplemented();
}

void WebProcessPool::platformInitializeWebProcess(const WebProcessProxy&, WebProcessCreationParameters&)
{
    notImplemented();
}

void WebProcessPool::platformInvalidateContext()
{
    notImplemented();
}

void WebProcessPool::platformResolvePathsForSandboxExtensions()
{
}

} // namespace WebKit
