/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 12, 2022.
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
#include "RemoteTextTrackProxy.h"

#if ENABLE(GPU_PROCESS) && ENABLE(VIDEO)

#include "Connection.h"
#include "GPUConnectionToWebProcess.h"
#include "MediaPlayerPrivateRemoteMessages.h"
#include "RemoteMediaPlayerProxy.h"
#include "TextTrackPrivateRemoteConfiguration.h"
#include <WebCore/ISOVTTCue.h>
#include <WebCore/NotImplemented.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteTextTrackProxy);

using namespace WebCore;

RemoteTextTrackProxy::RemoteTextTrackProxy(GPUConnectionToWebProcess& connectionToWebProcess, InbandTextTrackPrivate& trackPrivate, MediaPlayerIdentifier mediaPlayerIdentifier)
    : m_connectionToWebProcess(connectionToWebProcess)
    , m_trackPrivate(trackPrivate)
    , m_id(trackPrivate.id())
    , m_mediaPlayerIdentifier(mediaPlayerIdentifier)
{
    m_clientId = trackPrivate.addClient([](auto&& task) {
        ensureOnMainThread(WTFMove(task));
    }, *this);
    connectionToWebProcess.protectedConnection()->send(Messages::MediaPlayerPrivateRemote::AddRemoteTextTrack(configuration()), m_mediaPlayerIdentifier);
}

RemoteTextTrackProxy::~RemoteTextTrackProxy()
{
    Ref { m_trackPrivate }->removeClient(m_clientId);
}

TextTrackPrivateRemoteConfiguration& RemoteTextTrackProxy::configuration()
{
    static NeverDestroyed<TextTrackPrivateRemoteConfiguration> configuration;

    Ref trackPrivate = m_trackPrivate;
    configuration->trackId = trackPrivate->id();
    configuration->label = trackPrivate->label();
    configuration->language = trackPrivate->language();
    configuration->trackIndex = trackPrivate->trackIndex();
    configuration->inBandMetadataTrackDispatchType = trackPrivate->inBandMetadataTrackDispatchType();
    configuration->startTimeVariance = trackPrivate->startTimeVariance();

    configuration->cueFormat = trackPrivate->cueFormat();
    configuration->isClosedCaptions = trackPrivate->isClosedCaptions();
    configuration->isSDH = trackPrivate->isSDH();
    configuration->containsOnlyForcedSubtitles = trackPrivate->containsOnlyForcedSubtitles();
    configuration->isMainProgramContent = trackPrivate->isMainProgramContent();
    configuration->isEasyToRead = trackPrivate->isEasyToRead();
    configuration->isDefault = trackPrivate->isDefault();
    configuration->kind = trackPrivate->kind();

    return configuration.get();
}

void RemoteTextTrackProxy::configurationChanged()
{
    if (RefPtr connection = m_connectionToWebProcess.get())
        connection->protectedConnection()->send(Messages::MediaPlayerPrivateRemote::RemoteTextTrackConfigurationChanged(std::exchange(m_id, m_trackPrivate->id()), configuration()), m_mediaPlayerIdentifier);
}

void RemoteTextTrackProxy::willRemove()
{
    ASSERT_NOT_REACHED();
}

void RemoteTextTrackProxy::idChanged(TrackID)
{
    configurationChanged();
}

void RemoteTextTrackProxy::labelChanged(const AtomString&)
{
    configurationChanged();
}

void RemoteTextTrackProxy::languageChanged(const AtomString&)
{
    configurationChanged();
}

void RemoteTextTrackProxy::addDataCue(const MediaTime& start, const MediaTime& end, std::span<const uint8_t> data)
{
    if (RefPtr connection = m_connectionToWebProcess.get())
        connection->protectedConnection()->send(Messages::MediaPlayerPrivateRemote::AddDataCue(m_trackPrivate->id(), start, end, data), m_mediaPlayerIdentifier);
}

#if ENABLE(DATACUE_VALUE)
void RemoteTextTrackProxy::addDataCue(const MediaTime& start, const MediaTime& end, Ref<SerializedPlatformDataCue>&& cueData, const String& type)
{
    if (RefPtr connection = m_connectionToWebProcess.get())
        connection->protectedConnection()->send(Messages::MediaPlayerPrivateRemote::AddDataCueWithType(m_trackPrivate->id(), start, end, cueData->encodableValue(), type), m_mediaPlayerIdentifier);
}

void RemoteTextTrackProxy::updateDataCue(const MediaTime& start, const MediaTime& end, SerializedPlatformDataCue& cueData)
{
    if (RefPtr connection = m_connectionToWebProcess.get())
        connection->protectedConnection()->send(Messages::MediaPlayerPrivateRemote::UpdateDataCue(m_trackPrivate->id(), start, end, cueData.encodableValue()), m_mediaPlayerIdentifier);
}

void RemoteTextTrackProxy::removeDataCue(const MediaTime& start, const MediaTime& end, SerializedPlatformDataCue& cueData)
{
    if (RefPtr connection = m_connectionToWebProcess.get())
        connection->protectedConnection()->send(Messages::MediaPlayerPrivateRemote::RemoveDataCue(m_trackPrivate->id(), start, end, cueData.encodableValue()), m_mediaPlayerIdentifier);
}
#endif

void RemoteTextTrackProxy::addGenericCue(InbandGenericCue& cue)
{
    if (RefPtr connection = m_connectionToWebProcess.get())
        connection->protectedConnection()->send(Messages::MediaPlayerPrivateRemote::AddGenericCue(m_trackPrivate->id(), cue.cueData()), m_mediaPlayerIdentifier);
}

void RemoteTextTrackProxy::updateGenericCue(InbandGenericCue& cue)
{
    if (RefPtr connection = m_connectionToWebProcess.get())
        connection->protectedConnection()->send(Messages::MediaPlayerPrivateRemote::UpdateGenericCue(m_trackPrivate->id(), cue.cueData()), m_mediaPlayerIdentifier);
}

void RemoteTextTrackProxy::removeGenericCue(InbandGenericCue& cue)
{
    if (RefPtr connection = m_connectionToWebProcess.get())
        connection->protectedConnection()->send(Messages::MediaPlayerPrivateRemote::RemoveGenericCue(m_trackPrivate->id(), cue.cueData()), m_mediaPlayerIdentifier);
}

void RemoteTextTrackProxy::parseWebVTTFileHeader(String&& header)
{
    if (RefPtr connection = m_connectionToWebProcess.get())
        connection->protectedConnection()->send(Messages::MediaPlayerPrivateRemote::ParseWebVTTFileHeader(m_trackPrivate->id(), header), m_mediaPlayerIdentifier);
}

void RemoteTextTrackProxy::parseWebVTTCueData(std::span<const uint8_t> data)
{
    if (RefPtr connection = m_connectionToWebProcess.get())
        connection->protectedConnection()->send(Messages::MediaPlayerPrivateRemote::ParseWebVTTCueData(m_trackPrivate->id(), data), m_mediaPlayerIdentifier);
}

void RemoteTextTrackProxy::parseWebVTTCueData(ISOWebVTTCue&& cueData)
{
    if (RefPtr connection = m_connectionToWebProcess.get())
        connection->protectedConnection()->send(Messages::MediaPlayerPrivateRemote::ParseWebVTTCueDataStruct(m_trackPrivate->id(), cueData), m_mediaPlayerIdentifier);
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(VIDEO)
