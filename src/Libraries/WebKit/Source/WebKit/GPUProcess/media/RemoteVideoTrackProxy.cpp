/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 30, 2021.
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
#include "RemoteVideoTrackProxy.h"

#if ENABLE(GPU_PROCESS) && ENABLE(VIDEO)

#include "Connection.h"
#include "GPUConnectionToWebProcess.h"
#include "MediaPlayerPrivateRemoteMessages.h"
#include "RemoteMediaPlayerProxy.h"
#include "VideoTrackPrivateRemoteConfiguration.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteVideoTrackProxy);

using namespace WebCore;

RemoteVideoTrackProxy::RemoteVideoTrackProxy(GPUConnectionToWebProcess& connectionToWebProcess, VideoTrackPrivate& trackPrivate, MediaPlayerIdentifier mediaPlayerIdentifier)
    : m_connectionToWebProcess(connectionToWebProcess)
    , m_trackPrivate(trackPrivate)
    , m_id(trackPrivate.id())
    , m_mediaPlayerIdentifier(mediaPlayerIdentifier)
{
    m_clientRegistrationId = trackPrivate.addClient([](auto&& task) {
        ensureOnMainThread(WTFMove(task));
    }, *this);
    connectionToWebProcess.protectedConnection()->send(Messages::MediaPlayerPrivateRemote::AddRemoteVideoTrack(configuration()), m_mediaPlayerIdentifier);
}

RemoteVideoTrackProxy::~RemoteVideoTrackProxy()
{
    Ref { m_trackPrivate }->removeClient(m_clientRegistrationId);
}

VideoTrackPrivateRemoteConfiguration RemoteVideoTrackProxy::configuration()
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
        trackPrivate->selected(),
        trackPrivate->kind(),
        trackPrivate->configuration(),
    };
}

void RemoteVideoTrackProxy::updateConfiguration()
{
    if (RefPtr connection = m_connectionToWebProcess.get())
        connection->protectedConnection()->send(Messages::MediaPlayerPrivateRemote::RemoteVideoTrackConfigurationChanged(std::exchange(m_id, m_trackPrivate->id()), configuration()), m_mediaPlayerIdentifier);
}

void RemoteVideoTrackProxy::willRemove()
{
    ASSERT_NOT_REACHED();
}

void RemoteVideoTrackProxy::selectedChanged(bool selected)
{
    if (m_selected == selected)
        return;
    m_selected = selected;
    updateConfiguration();
}

void RemoteVideoTrackProxy::idChanged(TrackID)
{
    updateConfiguration();
}

void RemoteVideoTrackProxy::labelChanged(const AtomString&)
{
    updateConfiguration();
}

void RemoteVideoTrackProxy::languageChanged(const AtomString&)
{
    updateConfiguration();
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(VIDEO)
