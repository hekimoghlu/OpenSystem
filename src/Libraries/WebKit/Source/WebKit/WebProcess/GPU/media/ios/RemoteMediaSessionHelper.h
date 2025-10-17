/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 28, 2022.
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

#if ENABLE(GPU_PROCESS) && PLATFORM(IOS_FAMILY)

#include "GPUProcessConnection.h"
#include "MessageReceiver.h"
#include <WebCore/MediaPlaybackTargetContext.h>
#include <WebCore/MediaSessionHelperIOS.h>

namespace WebKit {

class MediaPlaybackTargetContextSerialized;
class WebProcess;

class RemoteMediaSessionHelper final
    : public WebCore::MediaSessionHelper
    , public IPC::MessageReceiver
    , public GPUProcessConnection::Client {
public:
    RemoteMediaSessionHelper();
    virtual ~RemoteMediaSessionHelper() = default;

    IPC::Connection& ensureConnection();

    using HasAvailableTargets = WebCore::MediaSessionHelperClient::HasAvailableTargets;
    using PlayingToAutomotiveHeadUnit = WebCore::MediaSessionHelperClient::PlayingToAutomotiveHeadUnit;
    using ShouldPause = WebCore::MediaSessionHelperClient::ShouldPause;
    using SupportsAirPlayVideo = WebCore::MediaSessionHelperClient::SupportsAirPlayVideo;
    using SuspendedUnderLock = WebCore::MediaSessionHelperClient::SuspendedUnderLock;

    WTF_ABSTRACT_THREAD_SAFE_REF_COUNTED_AND_CAN_MAKE_WEAK_PTR_IMPL;

private:
    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    // GPUProcessConnection::Client
    void gpuProcessConnectionDidClose(GPUProcessConnection&) final;

    // MediaSessionHelper
    void startMonitoringWirelessRoutesInternal() final;
    void stopMonitoringWirelessRoutesInternal() final;
    void providePresentingApplicationPID(int, ShouldOverride) final;

    // Messages
    void activeVideoRouteDidChange(SupportsAirPlayVideo, MediaPlaybackTargetContextSerialized&&);
    void activeAudioRouteSupportsSpatialPlaybackDidChange(SupportsSpatialAudioPlayback);

    ThreadSafeWeakPtr<GPUProcessConnection> m_gpuProcessConnection;
};

}

#endif
