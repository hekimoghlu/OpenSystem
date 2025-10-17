/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 18, 2022.
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
#include <wtf/MediaTime.h>

namespace WebCore {

class TextTrack;

class TextTrackList final : public TrackListBase {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(TextTrackList);
public:
    static Ref<TextTrackList> create(ScriptExecutionContext* context)
    {
        auto list = adoptRef(*new TextTrackList(context));
        list->suspendIfNeeded();
        return list;
    }
    virtual ~TextTrackList();

    bool isSupportedPropertyIndex(unsigned index) const { return index < length(); }
    unsigned length() const override;
    int getTrackIndex(TextTrack&);
    int getTrackIndexRelativeToRenderedTracks(TextTrack&);
    bool contains(TrackBase&) const override;

    TextTrack* item(unsigned index) const;
    TextTrack* getTrackById(const AtomString&) const;
    TextTrack* getTrackById(TrackID) const;
    TextTrack* lastItem() const { return item(length() - 1); }

    void append(Ref<TextTrack>&&);
    void remove(TrackBase&, bool scheduleEvent = true) override;

    void setDuration(MediaTime duration) { m_duration = duration; }
    const MediaTime& duration() const { return m_duration; }

    // EventTarget
    enum EventTargetInterfaceType eventTargetInterface() const override;

private:
    TextTrackList(ScriptExecutionContext*);

    void invalidateTrackIndexesAfterTrack(TextTrack&);

    Vector<RefPtr<TrackBase>> m_addTrackTracks;
    Vector<RefPtr<TrackBase>> m_elementTracks;
    MediaTime m_duration;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::TextTrackList)
    static bool isType(const WebCore::TrackListBase& trackList) { return trackList.type() == WebCore::TrackListBase::TextTrackList; }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(VIDEO)
