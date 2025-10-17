/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 22, 2024.
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
#include "RemoteWCLayerTreeHostProxy.h"

#if USE(GRAPHICS_LAYER_WC)

#include "GPUConnectionToWebProcess.h"
#include "MessageSenderInlines.h"
#include "RemoteWCLayerTreeHostMessages.h"
#include "WCUpdateInfo.h"
#include "WebPage.h"
#include "WebProcess.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteWCLayerTreeHostProxy);

RemoteWCLayerTreeHostProxy::RemoteWCLayerTreeHostProxy(WebPage& page, bool usesOffscreenRendering)
    : m_page(page)
    , m_usesOffscreenRendering(usesOffscreenRendering)
{
}

RemoteWCLayerTreeHostProxy::~RemoteWCLayerTreeHostProxy()
{
    disconnectGpuProcessIfNeeded();
}

IPC::Connection* RemoteWCLayerTreeHostProxy::messageSenderConnection() const
{
    return &const_cast<RemoteWCLayerTreeHostProxy&>(*this).ensureGPUProcessConnection().connection();
}

GPUProcessConnection& RemoteWCLayerTreeHostProxy::ensureGPUProcessConnection()
{
    auto gpuProcessConnection = m_gpuProcessConnection.get();
    if (!gpuProcessConnection) {
        gpuProcessConnection = &WebProcess::singleton().ensureGPUProcessConnection();
        m_gpuProcessConnection = gpuProcessConnection;
        gpuProcessConnection->addClient(*this);
        gpuProcessConnection->connection().send(
            Messages::GPUConnectionToWebProcess::CreateWCLayerTreeHost(wcLayerTreeHostIdentifier(), m_page->nativeWindowHandle(), m_usesOffscreenRendering),
            0, IPC::SendOption::DispatchMessageEvenWhenWaitingForSyncReply);
    }
    return *gpuProcessConnection;
}

void RemoteWCLayerTreeHostProxy::disconnectGpuProcessIfNeeded()
{
    if (auto gpuProcessConnection = std::exchange(m_gpuProcessConnection, nullptr).get()) {
        gpuProcessConnection->connection().send(Messages::GPUConnectionToWebProcess::ReleaseWCLayerTreeHost(wcLayerTreeHostIdentifier()), 0, IPC::SendOption::DispatchMessageEvenWhenWaitingForSyncReply);
    }
}

void RemoteWCLayerTreeHostProxy::gpuProcessConnectionDidClose(GPUProcessConnection& previousConnection)
{
    m_gpuProcessConnection = nullptr;
}

uint64_t RemoteWCLayerTreeHostProxy::messageSenderDestinationID() const
{
    return wcLayerTreeHostIdentifier().toUInt64();
}

void RemoteWCLayerTreeHostProxy::update(WCUpdateInfo&& updateInfo, CompletionHandler<void(std::optional<WebKit::UpdateInfo>)>&& completionHandler)
{
    sendWithAsyncReply(Messages::RemoteWCLayerTreeHost::Update(updateInfo), WTFMove(completionHandler));
}

} // namespace WebKit

#endif // USE(GRAPHICS_LAYER_WC)
