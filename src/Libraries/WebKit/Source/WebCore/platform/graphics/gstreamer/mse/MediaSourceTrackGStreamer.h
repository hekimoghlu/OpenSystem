/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 18, 2022.
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

#if ENABLE(MEDIA_SOURCE) && USE(GSTREAMER)

#include "TrackPrivateBaseGStreamer.h"
#include "TrackQueue.h"
#include <wtf/DataMutex.h>

namespace WebCore {

class MediaSourceTrackGStreamer final: public ThreadSafeRefCounted<MediaSourceTrackGStreamer> {
public:
    static Ref<MediaSourceTrackGStreamer> create(TrackPrivateBaseGStreamer::TrackType, TrackID, GRefPtr<GstCaps>&& initialCaps);
    virtual ~MediaSourceTrackGStreamer();

    TrackPrivateBaseGStreamer::TrackType type() const { return m_type; }
    TrackID id() const { return m_id; }
    GRefPtr<GstCaps>& initialCaps() { return m_initialCaps; }
    DataMutex<TrackQueue>& queueDataMutex() { return m_queueDataMutex; }

    bool isReadyForMoreSamples();

    void notifyWhenReadyForMoreSamples(TrackQueue::LowLevelHandler&&);

    void enqueueObject(GRefPtr<GstMiniObject>&&);

    // This method is provided to clear the TrackQueue in cases where the stream hasn't been started (e.g. because
    // another SourceBuffer hasn't received the necessary initalization segment for playback).
    // Otherwise, webKitMediaSrcFlush() should be used instead, which will also do a GStreamer pipeline flush where
    // necessary.
    void clearQueue();

    void remove();

private:
    explicit MediaSourceTrackGStreamer(TrackPrivateBaseGStreamer::TrackType, TrackID, GRefPtr<GstCaps>&& initialCaps);

    TrackPrivateBaseGStreamer::TrackType m_type;
    TrackID m_id;
    GRefPtr<GstCaps> m_initialCaps;
    DataMutex<TrackQueue> m_queueDataMutex;

    bool m_isRemoved { false };
};

}

#endif
