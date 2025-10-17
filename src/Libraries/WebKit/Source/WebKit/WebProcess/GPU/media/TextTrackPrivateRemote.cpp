/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 18, 2022.
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
#include "TextTrackPrivateRemote.h"

#if ENABLE(GPU_PROCESS) && ENABLE(VIDEO)

#include "GPUProcessConnection.h"
#include "MediaPlayerPrivateRemote.h"
#include "RemoteMediaPlayerProxyMessages.h"
#include <wtf/CrossThreadCopier.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(TextTrackPrivateRemote);

using namespace WebCore;

TextTrackPrivateRemote::TextTrackPrivateRemote(GPUProcessConnection& gpuProcessConnection, MediaPlayerIdentifier playerIdentifier, TextTrackPrivateRemoteConfiguration&& configuration)
    : WebCore::InbandTextTrackPrivate(configuration.cueFormat)
    , m_gpuProcessConnection(gpuProcessConnection)
    , m_id(configuration.trackId)
    , m_playerIdentifier(playerIdentifier)
{
    updateConfiguration(WTFMove(configuration));
}

void TextTrackPrivateRemote::setMode(TextTrackMode mode)
{
    auto gpuProcessConnection = m_gpuProcessConnection.get();
    if (!gpuProcessConnection)
        return;

    if (mode == InbandTextTrackPrivate::mode())
        return;

    gpuProcessConnection->connection().send(Messages::RemoteMediaPlayerProxy::TextTrackSetMode(m_id, mode), m_playerIdentifier);
    InbandTextTrackPrivate::setMode(mode);
}

void TextTrackPrivateRemote::updateConfiguration(TextTrackPrivateRemoteConfiguration&& configuration)
{
    if (configuration.trackId != m_id) {
        m_id = configuration.trackId;
        notifyClients([id = m_id](auto& client) {
            client.idChanged(id);
        });
    }

    if (configuration.label != m_label) {
        auto changed = !m_label.isEmpty();
        m_label = configuration.label;
        if (changed) {
            notifyClients([label = crossThreadCopy(m_label)](auto& client) {
                client.labelChanged(AtomString { label });
            });
        }
    }

    if (configuration.language != m_language) {
        auto changed = !m_language.isEmpty();
        m_language = configuration.language;
        if (changed) {
            notifyClients([language = crossThreadCopy(m_language)](auto& client) {
                client.languageChanged(AtomString { language });
            });
        }
    }

    m_trackIndex = configuration.trackIndex;
    m_inBandMetadataTrackDispatchType = configuration.inBandMetadataTrackDispatchType;
    m_startTimeVariance = configuration.startTimeVariance;

    m_kind = configuration.kind;
    m_isClosedCaptions = configuration.isClosedCaptions;
    m_isSDH = configuration.isSDH;
    m_containsOnlyForcedSubtitles = configuration.containsOnlyForcedSubtitles;
    m_isMainProgramContent = configuration.isMainProgramContent;
    m_isEasyToRead = configuration.isEasyToRead;
    m_isDefault = configuration.isDefault;
}

void TextTrackPrivateRemote::addGenericCue(Ref<InbandGenericCue> cue)
{
    ASSERT(hasClients());
    notifyClients([cue](auto& client) {
        downcast<InbandTextTrackPrivateClient>(client).addGenericCue(cue);
    });
}

void TextTrackPrivateRemote::updateGenericCue(Ref<InbandGenericCue> cue)
{
    ASSERT(hasClients());
    notifyClients([cue](auto& client) {
        downcast<InbandTextTrackPrivateClient>(client).updateGenericCue(cue);
    });
}

void TextTrackPrivateRemote::removeGenericCue(Ref<InbandGenericCue> cue)
{
    ASSERT(hasClients());
    notifyClients([cue](auto& client) {
        downcast<InbandTextTrackPrivateClient>(client).removeGenericCue(cue);
    });
}

void TextTrackPrivateRemote::parseWebVTTFileHeader(String&& header)
{
    ASSERT(hasOneClient());
    notifyMainThreadClient([&](auto& client) {
        downcast<InbandTextTrackPrivateClient>(client).parseWebVTTFileHeader(WTFMove(header));
    });
}

void TextTrackPrivateRemote::parseWebVTTCueData(std::span<const uint8_t> data)
{
    ASSERT(hasOneClient());
    notifyMainThreadClient([&](auto& client) {
        downcast<InbandTextTrackPrivateClient>(client).parseWebVTTCueData(data);
    });
}

void TextTrackPrivateRemote::parseWebVTTCueDataStruct(ISOWebVTTCue&& cueData)
{
    ASSERT(hasOneClient());
    notifyMainThreadClient([&](auto& client) {
        downcast<InbandTextTrackPrivateClient>(client).parseWebVTTCueData(WTFMove(cueData));
    });
}

void TextTrackPrivateRemote::addDataCue(MediaTime&& start, MediaTime&& end, std::span<const uint8_t> data)
{
    ASSERT(hasOneClient());
    notifyMainThreadClient([&](auto& client) {
        downcast<InbandTextTrackPrivateClient>(client).addDataCue(WTFMove(start), WTFMove(end), data);
    });
}

#if ENABLE(DATACUE_VALUE)
void TextTrackPrivateRemote::addDataCueWithType(MediaTime&& start, MediaTime&& end, SerializedPlatformDataCueValue&& dataValue, String&& type)
{
    ASSERT(hasOneClient());
    notifyMainThreadClient([&](auto& client) {
        downcast<InbandTextTrackPrivateClient>(client).addDataCue(WTFMove(start), WTFMove(end), WebCore::SerializedPlatformDataCue::create(WTFMove(dataValue)), type);
    });
}

void TextTrackPrivateRemote::updateDataCue(MediaTime&& start, MediaTime&& end, SerializedPlatformDataCueValue&& dataValue)
{
    ASSERT(hasOneClient());
    notifyMainThreadClient([&](auto& client) {
        downcast<InbandTextTrackPrivateClient>(client).updateDataCue(WTFMove(start), WTFMove(end), WebCore::SerializedPlatformDataCue::create(WTFMove(dataValue)));
    });
}

void TextTrackPrivateRemote::removeDataCue(MediaTime&& start, MediaTime&& end, SerializedPlatformDataCueValue&& dataValue)
{
    ASSERT(hasOneClient());
    notifyMainThreadClient([&](auto& client) {
        downcast<InbandTextTrackPrivateClient>(client).removeDataCue(WTFMove(start), WTFMove(end), WebCore::SerializedPlatformDataCue::create(WTFMove(dataValue)));
    });
}
#endif

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(VIDEO)
