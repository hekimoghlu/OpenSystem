/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 15, 2022.
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
#include "RemoteMediaSessionHelperProxy.h"

#if ENABLE(GPU_PROCESS) && PLATFORM(IOS_FAMILY)

#include "Connection.h"
#include "GPUConnectionToWebProcess.h"
#include "MediaPlaybackTargetContextSerialized.h"
#include "RemoteMediaSessionHelperMessages.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteMediaSessionHelperProxy);

RemoteMediaSessionHelperProxy::RemoteMediaSessionHelperProxy(GPUConnectionToWebProcess& gpuConnection)
    : m_gpuConnection(gpuConnection)
{
    MediaSessionHelper::sharedHelper().addClient(*this);
}

RemoteMediaSessionHelperProxy::~RemoteMediaSessionHelperProxy()
{
    stopMonitoringWirelessRoutes();
    MediaSessionHelper::sharedHelper().removeClient(*this);
}

void RemoteMediaSessionHelperProxy::ref() const
{
    m_gpuConnection.get()->ref();
}

void RemoteMediaSessionHelperProxy::deref() const
{
    m_gpuConnection.get()->deref();
}

void RemoteMediaSessionHelperProxy::startMonitoringWirelessRoutes()
{
    if (m_isMonitoringWirelessRoutes)
        return;

    m_isMonitoringWirelessRoutes = true;
    MediaSessionHelper::sharedHelper().startMonitoringWirelessRoutes();
}

void RemoteMediaSessionHelperProxy::stopMonitoringWirelessRoutes()
{
    if (!m_isMonitoringWirelessRoutes)
        return;

    m_isMonitoringWirelessRoutes = false;
    MediaSessionHelper::sharedHelper().stopMonitoringWirelessRoutes();
}

void RemoteMediaSessionHelperProxy::providePresentingApplicationPID(int pid, MediaSessionHelper::ShouldOverride shouldOverride)
{
    m_presentingApplicationPID = pid;
    MediaSessionHelper::sharedHelper().providePresentingApplicationPID(pid, shouldOverride);
}

void RemoteMediaSessionHelperProxy::overridePresentingApplicationPIDIfNeeded()
{
    if (m_presentingApplicationPID)
        MediaSessionHelper::sharedHelper().providePresentingApplicationPID(*m_presentingApplicationPID, MediaSessionHelper::ShouldOverride::Yes);
}

void RemoteMediaSessionHelperProxy::applicationWillEnterForeground(SuspendedUnderLock suspendedUnderLock)
{
    if (auto connection = m_gpuConnection.get())
        connection->connection().send(Messages::RemoteMediaSessionHelper::ApplicationWillEnterForeground(suspendedUnderLock), { });
}

void RemoteMediaSessionHelperProxy::applicationDidEnterBackground(SuspendedUnderLock suspendedUnderLock)
{
    if (auto connection = m_gpuConnection.get())
        connection->connection().send(Messages::RemoteMediaSessionHelper::ApplicationDidEnterBackground(suspendedUnderLock), { });
}

void RemoteMediaSessionHelperProxy::applicationWillBecomeInactive()
{
    if (auto connection = m_gpuConnection.get())
        connection->connection().send(Messages::RemoteMediaSessionHelper::ApplicationWillBecomeInactive(), { });
}

void RemoteMediaSessionHelperProxy::applicationDidBecomeActive()
{
    if (auto connection = m_gpuConnection.get())
        connection->connection().send(Messages::RemoteMediaSessionHelper::ApplicationDidBecomeActive(), { });
}

void RemoteMediaSessionHelperProxy::externalOutputDeviceAvailableDidChange(HasAvailableTargets hasAvailableTargets)
{
    if (auto connection = m_gpuConnection.get())
        connection->connection().send(Messages::RemoteMediaSessionHelper::ExternalOutputDeviceAvailableDidChange(hasAvailableTargets), { });
}

void RemoteMediaSessionHelperProxy::isPlayingToAutomotiveHeadUnitDidChange(PlayingToAutomotiveHeadUnit playing)
{
    if (auto connection = m_gpuConnection.get())
        connection->connection().send(Messages::RemoteMediaSessionHelper::IsPlayingToAutomotiveHeadUnitDidChange(playing), { });
}

void RemoteMediaSessionHelperProxy::activeAudioRouteDidChange(ShouldPause shouldPause)
{
    if (auto connection = m_gpuConnection.get())
        connection->connection().send(Messages::RemoteMediaSessionHelper::ActiveAudioRouteDidChange(shouldPause), { });
}

void RemoteMediaSessionHelperProxy::activeVideoRouteDidChange(SupportsAirPlayVideo supportsAirPlayVideo, Ref<WebCore::MediaPlaybackTarget>&& target)
{
    if (auto connection = m_gpuConnection.get())
        connection->connection().send(Messages::RemoteMediaSessionHelper::ActiveVideoRouteDidChange(supportsAirPlayVideo, MediaPlaybackTargetContextSerialized { target->targetContext() }), { });
}

void RemoteMediaSessionHelperProxy::activeAudioRouteSupportsSpatialPlaybackDidChange(SupportsSpatialAudioPlayback supportsSpatialPlayback)
{
    if (auto connection = m_gpuConnection.get())
        connection->connection().send(Messages::RemoteMediaSessionHelper::ActiveAudioRouteSupportsSpatialPlaybackDidChange(supportsSpatialPlayback), { });
}

std::optional<SharedPreferencesForWebProcess> RemoteMediaSessionHelperProxy::sharedPreferencesForWebProcess() const
{
    if (RefPtr gpuConnectionToWebProcess = m_gpuConnection.get())
        return gpuConnectionToWebProcess->sharedPreferencesForWebProcess();

    return std::nullopt;
}

}

#endif
