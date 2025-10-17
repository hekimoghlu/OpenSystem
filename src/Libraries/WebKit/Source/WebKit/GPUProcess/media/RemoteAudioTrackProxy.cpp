/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 26, 2024.
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
#include "RemoteAudioTrackProxy.h"

#if ENABLE(GPU_PROCESS) && ENABLE(VIDEO)

#include "AudioTrackPrivateRemoteConfiguration.h"
#include "Connection.h"
#include "GPUConnectionToWebProcess.h"
#include "MediaPlayerPrivateRemoteMessages.h"
#include "RemoteMediaPlayerProxy.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteAudioTrackProxy);

using namespace WebCore;

RemoteAudioTrackProxy::RemoteAudioTrackProxy(GPUConnectionToWebProcess& connectionToWebProcess, AudioTrackPrivate& trackPrivate, MediaPlayerIdentifier mediaPlayerIdentifier)
    : m_connectionToWebProcess(connectionToWebProcess)
    , m_trackPrivate(trackPrivate)
    , m_id(trackPrivate.id())
    , m_mediaPlayerIdentifier(mediaPlayerIdentifier)
{
    m_clientId = trackPrivate.addClient([](auto&& task) {
        ensureOnMainThread(WTFMove(task));
    }, *this);

    connectionToWebProcess.protectedConnection()->send(Messages::MediaPlayerPrivateRemote::AddRemoteAudioTrack(configuration()), m_mediaPlayerIdentifier);
}

RemoteAudioTrackProxy::~RemoteAudioTrackProxy()
{
    Ref { m_trackPrivate }->removeClient(m_clientId);
}

AudioTrackPrivateRemoteConfiguration RemoteAudioTrackProxy::configuration()
{
    Ref trackPrivate = m_trackPrivate;
    return {
        {
            trackPrivate->id(),
            trackPrivate->label(),
            trackPrivate->language(),
            trackPrivate->startTimeVariance(),
            trackPrivate->trackIndex(),
        },
        trackPrivate->enabled(),
        trackPrivate->kind(),
        trackPrivate->configuration(),
    };
}

void RemoteAudioTrackProxy::configurationChanged()
{
    RefPtr connection = m_connectionToWebProcess.get();
    if (!connection)
        return;
    connection->protectedConnection()->send(Messages::MediaPlayerPrivateRemote::RemoteAudioTrackConfigurationChanged(std::exchange(m_id, m_trackPrivate->id()), configuration()), m_mediaPlayerIdentifier);
}

void RemoteAudioTrackProxy::willRemove()
{
    ASSERT_NOT_REACHED();
}

void RemoteAudioTrackProxy::enabledChanged(bool enabled)
{
    if (enabled == m_enabled)
        return;
    m_enabled = enabled;
    configurationChanged();
}

void RemoteAudioTrackProxy::configurationChanged(const PlatformAudioTrackConfiguration& configuration)
{
    configurationChanged();
}

void RemoteAudioTrackProxy::idChanged(TrackID)
{
    configurationChanged();
}

void RemoteAudioTrackProxy::labelChanged(const AtomString&)
{
    configurationChanged();
}

void RemoteAudioTrackProxy::languageChanged(const AtomString&)
{
    configurationChanged();
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(VIDEO)
