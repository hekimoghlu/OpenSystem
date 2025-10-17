/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 13, 2022.
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

#if ENABLE(VIDEO)

#include "AudioTrack.h"
#include "Event.h"
#include "TextTrack.h"
#include "VideoTrack.h"

namespace WebCore {

class TrackEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(TrackEvent);
public:
    virtual ~TrackEvent();

    static Ref<TrackEvent> create(const AtomString& type, CanBubble canBubble, IsCancelable cancelable, Ref<TrackBase>&& track)
    {
        return adoptRef(*new TrackEvent(type, canBubble, cancelable, WTFMove(track)));
    }

    using TrackEventTrack = std::variant<RefPtr<VideoTrack>, RefPtr<AudioTrack>, RefPtr<TextTrack>>;

    struct Init : public EventInit {
        std::optional<TrackEventTrack> track;
    };

    static Ref<TrackEvent> create(const AtomString& type, Init&& initializer, IsTrusted isTrusted = IsTrusted::No)
    {
        return adoptRef(*new TrackEvent(type, WTFMove(initializer), isTrusted));
    }

    std::optional<TrackEventTrack> track() const { return m_track; }

private:
    TrackEvent(const AtomString& type, CanBubble, IsCancelable, Ref<TrackBase>&&);
    TrackEvent(const AtomString& type, Init&& initializer, IsTrusted);

    std::optional<TrackEventTrack> m_track;
};

} // namespace WebCore

#endif // ENABLE(VIDEO)
