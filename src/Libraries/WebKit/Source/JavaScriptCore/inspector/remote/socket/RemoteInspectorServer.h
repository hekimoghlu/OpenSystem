/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 8, 2022.
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

#if ENABLE(REMOTE_INSPECTOR)

#include "RemoteInspector.h"
#include "RemoteInspectorSocketEndpoint.h"

namespace Inspector {

class RemoteInspectorServer final : public RemoteInspectorSocketEndpoint::Listener {
public:
    ~RemoteInspectorServer() final;

    JS_EXPORT_PRIVATE static RemoteInspectorServer& singleton();

    JS_EXPORT_PRIVATE bool start(const char* address, uint16_t port);
    JS_EXPORT_PRIVATE std::optional<uint16_t> getPort() const;
    bool isRunning() const { return !!m_server; }

private:
    friend class LazyNeverDestroyed<RemoteInspectorServer>;
    RemoteInspectorServer() { Socket::init(); }

    std::optional<ConnectionID> doAccept(RemoteInspectorSocketEndpoint&, PlatformSocketType) final;
    void didChangeStatus(RemoteInspectorSocketEndpoint&, ConnectionID, RemoteInspectorSocketEndpoint::Listener::Status) final { };

    std::optional<ConnectionID> m_server;
};

} // namespace Inspector

#endif // ENABLE(REMOTE_INSPECTOR)
