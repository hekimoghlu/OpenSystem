/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 4, 2024.
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

#include "TrackEvent.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(TrackEvent);

static inline std::optional<TrackEvent::TrackEventTrack> convertToTrackEventTrack(Ref<TrackBase>&& track)
{
    switch (track->type()) {
    case TrackBase::BaseTrack:
        return std::nullopt;
    case TrackBase::TextTrack:
        return TrackEvent::TrackEventTrack { RefPtr { uncheckedDowncast<TextTrack>(WTFMove(track)) } };
    case TrackBase::AudioTrack:
        return TrackEvent::TrackEventTrack { RefPtr { uncheckedDowncast<AudioTrack>(WTFMove(track)) } };
    case TrackBase::VideoTrack:
        return TrackEvent::TrackEventTrack { RefPtr { uncheckedDowncast<VideoTrack>(WTFMove(track)) } };
    }
    
    ASSERT_NOT_REACHED();
    return std::nullopt;
}

TrackEvent::TrackEvent(const AtomString& type, CanBubble canBubble, IsCancelable cancelable, Ref<TrackBase>&& track)
    : Event(EventInterfaceType::TrackEvent, type, canBubble, cancelable)
    , m_track(convertToTrackEventTrack(WTFMove(track)))
{
}

TrackEvent::TrackEvent(const AtomString& type, Init&& initializer, IsTrusted isTrusted)
    : Event(EventInterfaceType::TrackEvent, type, initializer, isTrusted)
    , m_track(WTFMove(initializer.track))
{
}

TrackEvent::~TrackEvent() = default;

} // namespace WebCore

#endif
