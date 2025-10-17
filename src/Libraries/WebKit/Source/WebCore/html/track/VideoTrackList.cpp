/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 23, 2023.
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
#include "VideoTrackList.h"

#if ENABLE(VIDEO)

#include "ContextDestructionObserverInlines.h"
#include "ScriptExecutionContext.h"
#include "VideoTrack.h"

namespace WebCore {

VideoTrackList::VideoTrackList(ScriptExecutionContext* context)
    : TrackListBase(context, TrackListBase::VideoTrackList)
{
}

VideoTrackList::~VideoTrackList() = default;

void VideoTrackList::append(Ref<VideoTrack>&& track)
{
    // Insert tracks in the media file order.
    size_t index = track->inbandTrackIndex();
    size_t insertionIndex;
    for (insertionIndex = 0; insertionIndex < m_inbandTracks.size(); ++insertionIndex) {
        auto& otherTrack = downcast<VideoTrack>(*m_inbandTracks[insertionIndex]);
        if (otherTrack.inbandTrackIndex() > index)
            break;
    }
    m_inbandTracks.insert(insertionIndex, track.ptr());

    if (!track->trackList())
        track->setTrackList(*this);

    scheduleAddTrackEvent(WTFMove(track));
}

VideoTrack* VideoTrackList::item(unsigned index) const
{
    if (index < m_inbandTracks.size())
        return downcast<VideoTrack>(m_inbandTracks[index].get());
    return nullptr;
}

VideoTrack* VideoTrackList::getTrackById(const AtomString& id) const
{
    for (auto& inbandTracks : m_inbandTracks) {
        auto& track = downcast<VideoTrack>(*inbandTracks);
        if (track.id() == id)
            return &track;
    }
    return nullptr;
}

VideoTrack* VideoTrackList::getTrackById(TrackID id) const
{
    for (auto& inbandTracks : m_inbandTracks) {
        auto& track = downcast<VideoTrack>(*inbandTracks);
        if (track.trackId() == id)
            return &track;
    }
    return nullptr;
}

int VideoTrackList::selectedIndex() const
{
    // 4.8.10.10.1 AudioTrackList and VideoTrackList objects
    // The VideoTrackList.selectedIndex attribute must return the index of the
    // currently selected track, if any. If the VideoTrackList object does not
    // currently represent any tracks, or if none of the tracks are selected,
    // it must instead return âˆ’1.
    for (unsigned i = 0; i < length(); ++i) {
        if (downcast<VideoTrack>(*m_inbandTracks[i]).selected())
            return i;
    }
    return -1;
}

VideoTrack* VideoTrackList::selectedItem() const
{
    auto selectedIndex = this->selectedIndex();
    if (selectedIndex < 0)
        return nullptr;

    return item(selectedIndex);
}

enum EventTargetInterfaceType VideoTrackList::eventTargetInterface() const
{
    return EventTargetInterfaceType::VideoTrackList;
}

} // namespace WebCore
#endif
