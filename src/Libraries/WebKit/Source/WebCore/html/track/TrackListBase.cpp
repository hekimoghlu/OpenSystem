/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 7, 2024.
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

#if ENABLE(VIDEO)

#include "TrackListBase.h"

#include "EventNames.h"
#include "ScriptExecutionContext.h"
#include "TrackEvent.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(TrackListBase);

TrackListBase::TrackListBase(ScriptExecutionContext* context, Type type)
    : ActiveDOMObject(context)
    , m_type(type)
{
}

TrackListBase::~TrackListBase() = default;

void TrackListBase::didMoveToNewDocument(Document& newDocument)
{
    ActiveDOMObject::didMoveToNewDocument(newDocument);
    for (RefPtr track : m_inbandTracks)
        track->didMoveToNewDocument(newDocument);
}

WebCoreOpaqueRoot TrackListBase::opaqueRoot()
{
    if (auto* rootObserver = m_opaqueRootObserver.get())
        return (*rootObserver)();
    return WebCoreOpaqueRoot { this };
}

unsigned TrackListBase::length() const
{
    return m_inbandTracks.size();
}

RefPtr<TrackBase> TrackListBase::find(TrackID trackID) const
{
    size_t index = m_inbandTracks.findIf([&](auto& value) {
        return value->trackId() == trackID;
    });
    if (index == notFound)
        return nullptr;
    return m_inbandTracks[index];
}

void TrackListBase::remove(TrackID trackID, bool scheduleEvent)
{
    if (RefPtr track = find(trackID))
        remove(*track, scheduleEvent);
}

void TrackListBase::remove(TrackBase& track, bool scheduleEvent)
{
    size_t index = m_inbandTracks.find(&track);
    if (index == notFound)
        return;

    if (track.trackList() == this)
        track.clearTrackList();

    Ref<TrackBase> trackRef = *m_inbandTracks[index];

    m_inbandTracks.remove(index);

    if (scheduleEvent)
        scheduleRemoveTrackEvent(WTFMove(trackRef));
}

bool TrackListBase::contains(TrackBase& track) const
{
    return m_inbandTracks.find(&track) != notFound;
}

bool TrackListBase::contains(TrackID trackID) const
{
    return find(trackID);
}

void TrackListBase::scheduleTrackEvent(const AtomString& eventName, Ref<TrackBase>&& track)
{
    queueTaskToDispatchEvent(*this, TaskSource::MediaElement, TrackEvent::create(eventName, Event::CanBubble::No, Event::IsCancelable::No, WTFMove(track)));
}

void TrackListBase::scheduleAddTrackEvent(Ref<TrackBase>&& track)
{
    // 4.8.10.5 Loading the media resource
    // ...
    // Fire a trusted event with the name addtrack, that does not bubble and is
    // not cancelable, and that uses the TrackEvent interface, with the track
    // attribute initialized to the new AudioTrack object, at this
    // AudioTrackList object.
    // ...
    // Fire a trusted event with the name addtrack, that does not bubble and is
    // not cancelable, and that uses the TrackEvent interface, with the track
    // attribute initialized to the new VideoTrack object, at this
    // VideoTrackList object.

    // 4.8.10.12.3 Sourcing out-of-band text tracks
    // 4.8.10.12.4 Text track API
    // ... then queue a task to fire an event with the name addtrack, that does not
    // bubble and is not cancelable, and that uses the TrackEvent interface, with
    // the track attribute initialized to the text track's TextTrack object, at
    // the media element's textTracks attribute's TextTrackList object.
    scheduleTrackEvent(eventNames().addtrackEvent, WTFMove(track));
}

void TrackListBase::scheduleRemoveTrackEvent(Ref<TrackBase>&& track)
{
    // 4.8.10.6 Offsets into the media resource
    // If at any time the user agent learns that an audio or video track has
    // ended and all media data relating to that track corresponds to parts of
    // the media timeline that are before the earliest possible position, the
    // user agent may queue a task to remove the track from the audioTracks
    // attribute's AudioTrackList object or the videoTracks attribute's
    // VideoTrackList object as appropriate and then fire a trusted event
    // with the name removetrack, that does not bubble and is not cancelable,
    // and that uses the TrackEvent interface, with the track attribute
    // initialized to the AudioTrack or VideoTrack object representing the
    // track, at the media element's aforementioned AudioTrackList or
    // VideoTrackList object.

    // 4.8.10.12.3 Sourcing out-of-band text tracks
    // When a track element's parent element changes and the old parent was a
    // media element, then the user agent must remove the track element's
    // corresponding text track from the media element's list of text tracks,
    // and then queue a task to fire a trusted event with the name removetrack,
    // that does not bubble and is not cancelable, and that uses the TrackEvent
    // interface, with the track attribute initialized to the text track's
    // TextTrack object, at the media element's textTracks attribute's
    // TextTrackList object.
    scheduleTrackEvent(eventNames().removetrackEvent, WTFMove(track));
}

void TrackListBase::scheduleChangeEvent()
{
    // 4.8.10.6 Offsets into the media resource
    // Whenever an audio track in an AudioTrackList is enabled or disabled, the
    // user agent must queue a task to fire a simple event named change at the
    // AudioTrackList object.
    // ...
    // Whenever a track in a VideoTrackList that was previously not selected is
    // selected, the user agent must queue a task to fire a simple event named
    // change at the VideoTrackList object.
    m_isChangeEventScheduled = true;
    queueTaskKeepingObjectAlive(*this, TaskSource::MediaElement, [this] {
        m_isChangeEventScheduled = false;
        dispatchEvent(Event::create(eventNames().changeEvent, Event::CanBubble::No, Event::IsCancelable::No));
    });
}

bool TrackListBase::isAnyTrackEnabled() const
{
    for (auto& track : m_inbandTracks) {
        if (track->enabled())
            return true;
    }
    return false;
}

} // namespace WebCore

#endif
