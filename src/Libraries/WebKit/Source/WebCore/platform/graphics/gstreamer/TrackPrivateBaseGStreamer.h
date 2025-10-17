/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 1, 2025.
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

#include "AbortableTaskQueue.h"
#include "GStreamerCommon.h"
#include "MainThreadNotifier.h"
#include <gst/gst.h>
#include <wtf/Lock.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class TrackPrivateBase;
using TrackID = uint64_t;

class TrackPrivateBaseGStreamer {
public:
    virtual ~TrackPrivateBaseGStreamer();

    enum TrackType {
        Audio,
        Video,
        Text,
        Unknown
    };

    GstPad* pad() const { return m_pad.get(); }
    void setPad(GRefPtr<GstPad>&&);

    virtual void disconnect();

    virtual void setActive(bool) { }

    unsigned index() { return m_index; };
    void setIndex(unsigned index) { m_index =  index; }

    GstStream* stream() const { return m_stream.get(); }

    // Used for MSE, where the initial caps of the pad are relevant for initializing the matching pad in the
    // playback pipeline.
    void setInitialCaps(GRefPtr<GstCaps>&& caps) { m_initialCaps = WTFMove(caps); }
    const GRefPtr<GstCaps>& initialCaps() { return m_initialCaps; }

    TrackID streamId() const { return m_id; }
    const AtomString& gstStreamId() const { return m_gstStreamId; }

    virtual void updateConfigurationFromCaps(GRefPtr<GstCaps>&&) { }

protected:
    TrackPrivateBaseGStreamer(TrackType, TrackPrivateBase*, unsigned index, GRefPtr<GstPad>&&, bool shouldHandleStreamStartEvent);
    TrackPrivateBaseGStreamer(TrackType, TrackPrivateBase*, unsigned index, GRefPtr<GstPad>&&, TrackID);
    TrackPrivateBaseGStreamer(TrackType, TrackPrivateBase*, unsigned index, GstStream*);

    void notifyTrackOfTagsChanged();
    void notifyTrackOfStreamChanged();

    GstObject* objectForLogging() const;

    virtual void tagsChanged(GRefPtr<GstTagList>&&) { }
    virtual void capsChanged(TrackID, GRefPtr<GstCaps>&&) { }
    void installUpdateConfigurationHandlers();
    virtual void updateConfigurationFromTags(GRefPtr<GstTagList>&&) { }

    static GRefPtr<GstTagList> getAllTags(const GRefPtr<GstPad>&);

    enum MainThreadNotification {
        TagsChanged = 1 << 1,
        NewSample = 1 << 2,
        StreamChanged = 1 << 3
    };

    Ref<MainThreadNotifier<MainThreadNotification>> m_notifier;
    unsigned m_index;
    AtomString m_label;
    AtomString m_language;
    AtomString m_gstStreamId;
    // Track ID parsed from stream-id.
    TrackID m_id;
    GRefPtr<GstPad> m_pad;
    GRefPtr<GstPad> m_bestUpstreamPad;
    GRefPtr<GstStream> m_stream;
    unsigned long m_eventProbe { 0 };
    GRefPtr<GstCaps> m_initialCaps;
    AbortableTaskQueue m_taskQueue;

    // Track ID inferred from container-specific-track-id tag.
    std::optional<TrackID> m_trackID;
    bool updateTrackIDFromTags(const GRefPtr<GstTagList>&);

private:
    bool getLanguageCode(GstTagList* tags, AtomString& value);
    static AtomString generateUniquePlaybin2StreamID(TrackType, unsigned index);
    static char prefixForType(TrackType);
    template<class StringType>
    bool getTag(GstTagList* tags, const gchar* tagName, StringType& value);

    void streamChanged();
    void tagsChanged();

    TrackType m_type;
    TrackPrivateBase* m_owner;
    Lock m_tagMutex;
    GRefPtr<GstTagList> m_tags;
    bool m_shouldUsePadStreamId { true };
    bool m_shouldHandleStreamStartEvent { true };
};

} // namespace WebCore

#endif // ENABLE(VIDEO) && USE(GSTREAMER)
