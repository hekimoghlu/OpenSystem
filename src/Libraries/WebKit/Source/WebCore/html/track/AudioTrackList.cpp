/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 19, 2023.
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
#include "AudioTrackList.h"

#if ENABLE(VIDEO)

#include "AudioTrack.h"
#include "ContextDestructionObserverInlines.h"

namespace WebCore {

AudioTrackList::AudioTrackList(ScriptExecutionContext* context)
    : TrackListBase(context, TrackListBase::VideoTrackList)
{
}

AudioTrackList::~AudioTrackList() = default;

void AudioTrackList::append(Ref<AudioTrack>&& track)
{
    // Insert tracks in the media file order.
    size_t index = track->inbandTrackIndex();
    size_t insertionIndex;
    for (insertionIndex = 0; insertionIndex < m_inbandTracks.size(); ++insertionIndex) {
        auto& otherTrack = downcast<AudioTrack>(*m_inbandTracks[insertionIndex]);
        if (otherTrack.inbandTrackIndex() > index)
            break;
    }
    m_inbandTracks.insert(insertionIndex, track.ptr());

    if (!track->trackList())
        track->setTrackList(*this);

    scheduleAddTrackEvent(WTFMove(track));
}

void AudioTrackList::remove(TrackBase& track, bool scheduleEvent)
{
    auto& audioTrack = downcast<AudioTrack>(track);
    if (audioTrack.trackList() == this)
        audioTrack.clearTrackList();

    TrackListBase::remove(track, scheduleEvent);
}

AudioTrack* AudioTrackList::item(unsigned index) const
{
    if (index < m_inbandTracks.size())
        return downcast<AudioTrack>(m_inbandTracks[index].get());
    return nullptr;
}

AudioTrack* AudioTrackList::firstEnabled() const
{
    for (auto& item : m_inbandTracks) {
        if (item && item->enabled())
            return downcast<AudioTrack>(item.get());
    }
    return nullptr;
}

AudioTrack* AudioTrackList::getTrackById(const AtomString& id) const
{
    for (auto& inbandTrack : m_inbandTracks) {
        auto& track = downcast<AudioTrack>(*inbandTrack);
        if (track.id() == id)
            return &track;
    }
    return nullptr;
}

AudioTrack* AudioTrackList::getTrackById(TrackID id) const
{
    for (auto& inbandTrack : m_inbandTracks) {
        auto& track = downcast<AudioTrack>(*inbandTrack);
        if (track.trackId() == id)
            return &track;
    }
    return nullptr;
}

enum EventTargetInterfaceType AudioTrackList::eventTargetInterface() const
{
    return EventTargetInterfaceType::AudioTrackList;
}

} // namespace WebCore
#endif
