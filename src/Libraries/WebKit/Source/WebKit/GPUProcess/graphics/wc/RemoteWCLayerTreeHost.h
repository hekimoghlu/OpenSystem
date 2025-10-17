/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 9, 2022.
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

#if USE(GRAPHICS_LAYER_WC)

#include "Connection.h"
#include "MessageReceiver.h"
#include "MessageSender.h"
#include "WCLayerTreeHostIdentifier.h"
#include "WCSharedSceneContextHolder.h"
#include <WebCore/ProcessIdentifier.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>

namespace IPC {
class StreamConnectionWorkQueue;
}

namespace WebKit {
class GPUConnectionToWebProcess;
class WCScene;
struct UpdateInfo;
struct WCUpdateInfo;

class RemoteWCLayerTreeHost : public IPC::MessageReceiver, private IPC::MessageSender, public RefCounted<RemoteWCLayerTreeHost> {
    WTF_MAKE_TZONE_ALLOCATED(RemoteWCLayerTreeHost);
public:
    static Ref<RemoteWCLayerTreeHost> create(GPUConnectionToWebProcess&, WebKit::WCLayerTreeHostIdentifier, uint64_t nativeWindow, bool usesOffscreenRendering);
    ~RemoteWCLayerTreeHost();
    // message handlers
    void update(WCUpdateInfo&&, CompletionHandler<void(std::optional<WebKit::UpdateInfo>)>&&);

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

private:
    RemoteWCLayerTreeHost(GPUConnectionToWebProcess&, WebKit::WCLayerTreeHostIdentifier, uint64_t nativeWindow, bool usesOffscreenRendering);

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;
    // IPC::MessageSender
    IPC::Connection* messageSenderConnection() const override;
    uint64_t messageSenderDestinationID() const override;

    WeakPtr<GPUConnectionToWebProcess> m_connectionToWebProcess;
    WebCore::ProcessIdentifier m_webProcessIdentifier;
    WCLayerTreeHostIdentifier m_identifier;
    RefPtr<WCSharedSceneContextHolder::Holder> m_sharedSceneContextHolder;
    std::unique_ptr<WCScene> m_scene;
};

IPC::StreamConnectionWorkQueue& remoteGraphicsStreamWorkQueue();

} // namespace WebKit

#endif // USE(GRAPHICS_LAYER_WC)
