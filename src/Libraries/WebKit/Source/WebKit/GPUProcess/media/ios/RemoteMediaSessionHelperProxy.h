/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 16, 2025.
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

#include "MessageReceiver.h"
#include <WebCore/MediaSessionHelperIOS.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/WeakRef.h>

namespace WebKit {

class GPUConnectionToWebProcess;

class RemoteMediaSessionHelperProxy
    : public WebCore::MediaSessionHelperClient
    , public IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(RemoteMediaSessionHelperProxy);
public:
    RemoteMediaSessionHelperProxy(GPUConnectionToWebProcess&);
    virtual ~RemoteMediaSessionHelperProxy();

    void didReceiveMessageFromWebProcess(IPC::Connection& connection, IPC::Decoder& decoder) { didReceiveMessage(connection, decoder); }

    void overridePresentingApplicationPIDIfNeeded();
    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const;

    void ref() const final;
    void deref() const final;

private:
    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    // Messages
    void startMonitoringWirelessRoutes();
    void stopMonitoringWirelessRoutes();
    void providePresentingApplicationPID(int, WebCore::MediaSessionHelper::ShouldOverride);

    // MediaSessionHelperClient
    void applicationWillEnterForeground(SuspendedUnderLock) final;
    void applicationDidEnterBackground(SuspendedUnderLock) final;
    void applicationWillBecomeInactive() final;
    void applicationDidBecomeActive() final;
    void externalOutputDeviceAvailableDidChange(HasAvailableTargets) final;
    void isPlayingToAutomotiveHeadUnitDidChange(PlayingToAutomotiveHeadUnit) final;
    void activeAudioRouteDidChange(ShouldPause) final;
    void activeVideoRouteDidChange(SupportsAirPlayVideo, Ref<WebCore::MediaPlaybackTarget>&&) final;
    void activeAudioRouteSupportsSpatialPlaybackDidChange(SupportsSpatialAudioPlayback) final;

    bool m_isMonitoringWirelessRoutes { false };
    ThreadSafeWeakPtr<GPUConnectionToWebProcess> m_gpuConnection;
    std::optional<int> m_presentingApplicationPID;
};

}

#endif
