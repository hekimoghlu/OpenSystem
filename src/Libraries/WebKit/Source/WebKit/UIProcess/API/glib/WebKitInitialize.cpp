/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 5, 2025.
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
#include "WebKitInitialize.h"

#include "RemoteInspectorHTTPServer.h"
#include "WebKit2Initialize.h"
#include <JavaScriptCore/RemoteInspector.h>
#include <JavaScriptCore/RemoteInspectorServer.h>
#include <limits>
#include <mutex>
#include <wtf/glib/GRefPtr.h>
#include <wtf/glib/GUniquePtr.h>

#if USE(SKIA)
#include <skia/core/SkGraphics.h>
#endif

#if USE(SYSPROF_CAPTURE)
#include <wtf/SystemTracing.h>
#endif

namespace WebKit {

#if ENABLE(REMOTE_INSPECTOR)
static void initializeRemoteInspectorServer()
{
    const char* address = g_getenv("WEBKIT_INSPECTOR_SERVER");
    const char* httpAddress = g_getenv("WEBKIT_INSPECTOR_HTTP_SERVER");
    if (!address && !httpAddress)
        return;

    if (Inspector::RemoteInspectorServer::singleton().isRunning())
        return;

    auto parseAddress = [](const char* address) -> GRefPtr<GSocketAddress> {
        if (!address || !address[0])
            return nullptr;

        GUniquePtr<char> inspectorAddress(g_strdup(address));

        WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN // GTK/WPE port
        char* portPtr = g_strrstr(inspectorAddress.get(), ":");
        if (!portPtr)
            return nullptr;

        *portPtr = '\0';
        portPtr++;
        auto port = g_ascii_strtoull(portPtr, nullptr, 10);
        if (!port || port > std::numeric_limits<uint16_t>::max())
            return nullptr;

        char* addressPtr = inspectorAddress.get();
        if (addressPtr[0] == '[' && *(portPtr - 2) == ']') {
            // Strip the square brackets.
            addressPtr++;
            *(portPtr - 2) = '\0';
        }
        WTF_ALLOW_UNSAFE_BUFFER_USAGE_END

        return adoptGRef(g_inet_socket_address_new_from_string(addressPtr, port));
    };

    auto inspectorHTTPAddress = parseAddress(httpAddress);
    GRefPtr<GSocketAddress> inspectorAddress;
    if (inspectorHTTPAddress)
        inspectorAddress = adoptGRef(G_SOCKET_ADDRESS(g_inet_socket_address_new(g_inet_socket_address_get_address(G_INET_SOCKET_ADDRESS(inspectorHTTPAddress.get())), 0)));
    else
        inspectorAddress = parseAddress(address);
    if (!inspectorHTTPAddress && !inspectorAddress) {
        g_warning("Failed to start remote inspector server on %s: invalid address", address ? address : httpAddress);
        return;
    }

    if (!Inspector::RemoteInspectorServer::singleton().start(WTFMove(inspectorAddress)))
        return;

    if (inspectorHTTPAddress) {
        if (RemoteInspectorHTTPServer::singleton().start(WTFMove(inspectorHTTPAddress), Inspector::RemoteInspectorServer::singleton().port()))
            Inspector::RemoteInspector::setInspectorServerAddress(RemoteInspectorHTTPServer::singleton().inspectorServerAddress().utf8());
    } else
        Inspector::RemoteInspector::setInspectorServerAddress(address);
}
#endif

void webkitInitialize()
{
    static std::once_flag onceFlag;

    std::call_once(onceFlag, [] {
#if USE(SYSPROF_CAPTURE)
        SysprofAnnotator::createIfNeeded("WebKit (UI)"_s);
#endif
        InitializeWebKit2();
#if USE(SKIA)
        SkGraphics::Init();
#endif
#if ENABLE(REMOTE_INSPECTOR)
        initializeRemoteInspectorServer();
#endif
    });
}

} // namespace WebKit
