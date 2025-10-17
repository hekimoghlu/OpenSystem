/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 10, 2022.
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

#if ENABLE(VIDEO) && USE(GSTREAMER)

#include "InbandTextTrackPrivate.h"
#include "TrackPrivateBaseGStreamer.h"
#include <wtf/Forward.h>

namespace WebCore {

class MediaPlayerPrivateGStreamer;

class InbandTextTrackPrivateGStreamer : public InbandTextTrackPrivate, public TrackPrivateBaseGStreamer {
public:
    static Ref<InbandTextTrackPrivateGStreamer> create(unsigned index, GRefPtr<GstPad>&& pad, bool shouldHandleStreamStartEvent = true)
    {
        return adoptRef(*new InbandTextTrackPrivateGStreamer(index, WTFMove(pad), shouldHandleStreamStartEvent));
    }

    static Ref<InbandTextTrackPrivateGStreamer> create(unsigned index, GRefPtr<GstPad>&& pad, TrackID trackId)
    {
        return adoptRef(*new InbandTextTrackPrivateGStreamer(index, WTFMove(pad), trackId));
    }

    static Ref<InbandTextTrackPrivateGStreamer> create(ThreadSafeWeakPtr<MediaPlayerPrivateGStreamer>&&, unsigned index, GRefPtr<GstPad> pad)
    {
        return create(index, WTFMove(pad));
    }

    static Ref<InbandTextTrackPrivateGStreamer> create(ThreadSafeWeakPtr<MediaPlayerPrivateGStreamer>&&, unsigned index, GstStream* stream)
    {
        return adoptRef(*new InbandTextTrackPrivateGStreamer(index, stream));
    }

    Kind kind() const final { return m_kind; }
    TrackID id() const final { return m_trackID.value_or(m_id); }
    std::optional<AtomString> trackUID() const final { return std::nullopt; }
    AtomString label() const final { return m_label; }
    AtomString language() const final { return m_language; }
    int trackIndex() const final { return m_index; }

    void handleSample(GRefPtr<GstSample>&&);

protected:
    void tagsChanged(GRefPtr<GstTagList>&&) final;

private:
    InbandTextTrackPrivateGStreamer(unsigned index, GRefPtr<GstPad>&&, bool shouldHandleStreamStartEvent);
    InbandTextTrackPrivateGStreamer(unsigned index, GRefPtr<GstPad>&&, TrackID);
    InbandTextTrackPrivateGStreamer(unsigned index, GstStream*);

    void notifyTrackOfSample();

    Vector<GRefPtr<GstSample>> m_pendingSamples WTF_GUARDED_BY_LOCK(m_sampleMutex);
    Kind m_kind;
    Lock m_sampleMutex;
};

} // namespace WebCore

#endif // ENABLE(VIDEO) && USE(GSTREAMER)
