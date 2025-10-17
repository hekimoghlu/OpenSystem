/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 19, 2025.
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
#include "RemoteMediaSessionHelper.h"

#if ENABLE(GPU_PROCESS) && PLATFORM(IOS_FAMILY)

#include "Connection.h"
#include "GPUConnectionToWebProcessMessages.h"
#include "MediaPlaybackTargetContextSerialized.h"
#include "RemoteMediaSessionHelperMessages.h"
#include "RemoteMediaSessionHelperProxyMessages.h"
#include "WebProcess.h"
#include <WebCore/MediaPlaybackTargetCocoa.h>

namespace WebKit {

using namespace WebCore;

RemoteMediaSessionHelper::RemoteMediaSessionHelper() = default;

IPC::Connection& RemoteMediaSessionHelper::ensureConnection()
{
    auto gpuProcessConnection = m_gpuProcessConnection.get();
    if (!gpuProcessConnection) {
        gpuProcessConnection = &WebProcess::singleton().ensureGPUProcessConnection();
        m_gpuProcessConnection = gpuProcessConnection;
        gpuProcessConnection->addClient(*this);
        gpuProcessConnection->messageReceiverMap().addMessageReceiver(Messages::RemoteMediaSessionHelper::messageReceiverName(), *this);
        gpuProcessConnection->connection().send(Messages::GPUConnectionToWebProcess::EnsureMediaSessionHelper(), { });
    }
    return gpuProcessConnection->connection();
}

void RemoteMediaSessionHelper::gpuProcessConnectionDidClose(GPUProcessConnection& gpuProcessConnection)
{
    gpuProcessConnection.messageReceiverMap().removeMessageReceiver(*this);
    m_gpuProcessConnection = nullptr;
}

void RemoteMediaSessionHelper::startMonitoringWirelessRoutesInternal()
{
    ensureConnection().send(Messages::RemoteMediaSessionHelperProxy::StartMonitoringWirelessRoutes(), { });
}

void RemoteMediaSessionHelper::stopMonitoringWirelessRoutesInternal()
{
    ensureConnection().send(Messages::RemoteMediaSessionHelperProxy::StopMonitoringWirelessRoutes(), { });
}

void RemoteMediaSessionHelper::providePresentingApplicationPID(int pid, ShouldOverride shouldOverride)
{
    ensureConnection().send(Messages::RemoteMediaSessionHelperProxy::ProvidePresentingApplicationPID(pid, shouldOverride), { });
}

void RemoteMediaSessionHelper::activeVideoRouteDidChange(SupportsAirPlayVideo supportsAirPlayVideo, MediaPlaybackTargetContextSerialized&& targetContext)
{
    WTF::switchOn(targetContext.platformContext(), [](WebCore::MediaPlaybackTargetContextMock&&) {
        return;
    }, [&](WebCore::MediaPlaybackTargetContextCocoa&& context) {
        WebCore::MediaSessionHelper::activeVideoRouteDidChange(supportsAirPlayVideo, WebCore::MediaPlaybackTargetCocoa::create(WTFMove(context)));
    });
}

void RemoteMediaSessionHelper::activeAudioRouteSupportsSpatialPlaybackDidChange(SupportsSpatialAudioPlayback supportsSpatialAudioPlayback)
{
    WebCore::MediaSessionHelper::activeAudioRouteSupportsSpatialPlaybackDidChange(supportsSpatialAudioPlayback);
}

}

#endif

