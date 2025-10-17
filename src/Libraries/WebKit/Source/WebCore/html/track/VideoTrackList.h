/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 17, 2025.
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

class VideoTrack;

class VideoTrackList final : public TrackListBase {
public:
    static Ref<VideoTrackList> create(ScriptExecutionContext* context)
    {
        auto list = adoptRef(*new VideoTrackList(context));
        list->suspendIfNeeded();
        return list;
    }
    virtual ~VideoTrackList();

    VideoTrack* getTrackById(const AtomString&) const;
    VideoTrack* getTrackById(TrackID) const;
    int selectedIndex() const;

    bool isSupportedPropertyIndex(unsigned index) const { return index < m_inbandTracks.size(); }
    VideoTrack* item(unsigned) const;
    VideoTrack* lastItem() const { return item(length() - 1); }
    VideoTrack* selectedItem() const;
    void append(Ref<VideoTrack>&&);

    // EventTarget
    enum EventTargetInterfaceType eventTargetInterface() const override;

private:
    VideoTrackList(ScriptExecutionContext*);
};
static_assert(sizeof(VideoTrackList) == sizeof(TrackListBase));

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::VideoTrackList)
    static bool isType(const WebCore::TrackListBase& trackList) { return trackList.type() == WebCore::TrackListBase::VideoTrackList; }
SPECIALIZE_TYPE_TRAITS_END()

#endif // ENABLE(VIDEO)
