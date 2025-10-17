/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 2, 2023.
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
#include "RemoteInspectorServer.h"

#if ENABLE(REMOTE_INSPECTOR)

#include "RemoteInspectorMessageParser.h"
#include <wtf/MainThread.h>
#include <wtf/NeverDestroyed.h>

namespace Inspector {

RemoteInspectorServer& RemoteInspectorServer::singleton()
{
    static LazyNeverDestroyed<RemoteInspectorServer> shared;
    static std::once_flag onceKey;
    std::call_once(onceKey, [&] {
        shared.construct();
    });
    return shared;
}

RemoteInspectorServer::~RemoteInspectorServer()
{
    auto& endpoint = Inspector::RemoteInspectorSocketEndpoint::singleton();
    endpoint.invalidateListener(*this);
}

bool RemoteInspectorServer::start(const char* address, uint16_t port)
{
    if (isRunning())
        return false;

    auto& endpoint = Inspector::RemoteInspectorSocketEndpoint::singleton();
    m_server = endpoint.listenInet(address, port, *this);
    return isRunning();
}

std::optional<uint16_t> RemoteInspectorServer::getPort() const
{
    if (!isRunning())
        return std::nullopt;

    const auto& endpoint = Inspector::RemoteInspectorSocketEndpoint::singleton();
    return endpoint.getPort(m_server.value());
}

std::optional<ConnectionID> RemoteInspectorServer::doAccept(RemoteInspectorSocketEndpoint& endpoint, PlatformSocketType socket)
{
    ASSERT(!isMainThread());

    auto& inspector = RemoteInspector::singleton();
    if (inspector.isConnected()) {
        LOG_ERROR("RemoteInspector can accept only 1 client");
        return std::nullopt;
    }

    if (auto newID = endpoint.createClient(socket, inspector)) {
        inspector.connect(newID.value());
        return newID;
    }

    return std::nullopt;
}

} // namespace Inspector

#endif // ENABLE(REMOTE_INSPECTOR)
