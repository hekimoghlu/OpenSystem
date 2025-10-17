/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 31, 2023.
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
#include "RemoteWCLayerTreeHost.h"

#if USE(GRAPHICS_LAYER_WC)

#include "GPUConnectionToWebProcess.h"
#include "GPUProcess.h"
#include "RemoteGraphicsContextGL.h"
#include "RemoteWCLayerTreeHostMessages.h"
#include "StreamConnectionWorkQueue.h"
#include "WCScene.h"
#include "WCUpdateInfo.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

IPC::StreamConnectionWorkQueue& remoteGraphicsStreamWorkQueue()
{
#if ENABLE(WEBGL)
    return remoteGraphicsContextGLStreamWorkQueueSingleton();
#else
    static LazyNeverDestroyed<IPC::StreamConnectionWorkQueue> instance;
    static std::once_flag onceKey;
    std::call_once(onceKey, [&] {
        instance.construct("RemoteWCLayerTreeHost work queue"_s); // LazyNeverDestroyed owns the initial ref.
    });
    return instance.get();
#endif
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteWCLayerTreeHost);

Ref<RemoteWCLayerTreeHost> RemoteWCLayerTreeHost::create(GPUConnectionToWebProcess& connectionToWebProcess, WebKit::WCLayerTreeHostIdentifier identifier, uint64_t nativeWindow, bool usesOffscreenRendering)
{
    return adoptRef(*new RemoteWCLayerTreeHost(connectionToWebProcess, identifier, nativeWindow, usesOffscreenRendering));
}

RemoteWCLayerTreeHost::RemoteWCLayerTreeHost(GPUConnectionToWebProcess& connectionToWebProcess, WebKit::WCLayerTreeHostIdentifier identifier, uint64_t nativeWindow, bool usesOffscreenRendering)
    : m_connectionToWebProcess(connectionToWebProcess)
    , m_webProcessIdentifier(connectionToWebProcess.webProcessIdentifier())
    , m_identifier(identifier)
    , m_sharedSceneContextHolder(connectionToWebProcess.gpuProcess().sharedSceneContext().ensureHolderForWindow(nativeWindow))
{
    m_connectionToWebProcess->messageReceiverMap().addMessageReceiver(Messages::RemoteWCLayerTreeHost::messageReceiverName(), m_identifier.toUInt64(), *this);
    m_scene = makeUnique<WCScene>(m_webProcessIdentifier, usesOffscreenRendering);

    remoteGraphicsStreamWorkQueue().dispatch([scene = m_scene.get(), sceneContextHolder = m_sharedSceneContextHolder.get(), nativeWindow] {
        if (!sceneContextHolder->context)
            sceneContextHolder->context.emplace(nativeWindow);
        scene->initialize(*sceneContextHolder->context);
    });
}

RemoteWCLayerTreeHost::~RemoteWCLayerTreeHost()
{
    ASSERT(m_connectionToWebProcess);
    m_connectionToWebProcess->messageReceiverMap().removeMessageReceiver(Messages::RemoteWCLayerTreeHost::messageReceiverName(), m_identifier.toUInt64());
    auto sceneContextHolder = m_connectionToWebProcess->gpuProcess().sharedSceneContext().removeHolder(m_sharedSceneContextHolder.releaseNonNull());

    remoteGraphicsStreamWorkQueue().dispatch([sceneContextHolder = WTFMove(sceneContextHolder), scene = WTFMove(m_scene)]() mutable {
        // Destroy scene on the StreamWorkQueue thread.
        scene = nullptr;
        // sceneContextHolder can be destroyed on the StreamWorkQueue thread because it hasOneRef.
    });
}

IPC::Connection* RemoteWCLayerTreeHost::messageSenderConnection() const
{
    return &m_connectionToWebProcess->connection();
}

uint64_t RemoteWCLayerTreeHost::messageSenderDestinationID() const
{
    return m_identifier.toUInt64();
}

void RemoteWCLayerTreeHost::update(WCUpdateInfo&& update, CompletionHandler<void(std::optional<WebKit::UpdateInfo>)>&& completionHandler)
{
    remoteGraphicsStreamWorkQueue().dispatch([scene = m_scene.get(), update = WTFMove(update), completionHandler = WTFMove(completionHandler)]() mutable {
        auto updateInfo = scene->update(WTFMove(update));
        RunLoop::main().dispatch([updateInfo = WTFMove(updateInfo), completionHandler = WTFMove(completionHandler)]() mutable {
            completionHandler(WTFMove(updateInfo));
        });
    });
}

} // namespace WebKit

#endif // USE(GRAPHICS_LAYER_WC)
