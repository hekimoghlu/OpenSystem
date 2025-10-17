/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 27, 2024.
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

#include "TrackListBase.h"

namespace WebCore {

class AudioTrack;

class AudioTrackList final : public TrackListBase {
public:
    static Ref<AudioTrackList> create(ScriptExecutionContext* context)
    {
        auto list = adoptRef(*new AudioTrackList(context));
        list->suspendIfNeeded();
        return list;
    }
    virtual ~AudioTrackList();

    AudioTrack* getTrackById(const AtomString&) const;
    AudioTrack* getTrackById(TrackID) const;

    bool isSupportedPropertyIndex(unsigned index) const { return index < m_inbandTracks.size(); }
    AudioTrack* item(unsigned index) const;
    AudioTrack* lastItem() const { return item(length() - 1); }
    AudioTrack* firstEnabled() const;
    void append(Ref<AudioTrack>&&);
    void remove(TrackBase&, bool scheduleEvent = true) final;

    // EventTarget
    enum EventTargetInterfaceType eventTargetInterface() const override;

private:
    AudioTrackList(ScriptExecutionContext*);
};
static_assert(sizeof(AudioTrackList) == sizeof(TrackListBase));

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::AudioTrackList)
    static bool isType(const WebCore::TrackListBase& trackList) { return trackList.type() == WebCore::TrackListBase::AudioTrackList; }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(VIDEO)
