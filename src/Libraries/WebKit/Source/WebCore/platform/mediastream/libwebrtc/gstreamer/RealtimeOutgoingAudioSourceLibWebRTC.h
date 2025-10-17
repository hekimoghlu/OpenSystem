/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 26, 2024.
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

#if USE(LIBWEBRTC)

#include "GStreamerCommon.h"
#include "RealtimeOutgoingAudioSource.h"

#include <gst/audio/audio.h>
#include <wtf/CheckedRef.h>
#include <wtf/Lock.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class RealtimeOutgoingAudioSourceLibWebRTC final : public RealtimeOutgoingAudioSource, public CanMakeCheckedPtr<RealtimeOutgoingAudioSourceLibWebRTC> {
    WTF_MAKE_TZONE_ALLOCATED(RealtimeOutgoingAudioSourceLibWebRTC);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RealtimeOutgoingAudioSourceLibWebRTC);
public:
    static Ref<RealtimeOutgoingAudioSourceLibWebRTC> create(Ref<MediaStreamTrackPrivate>&& audioTrackPrivate)
    {
        return adoptRef(*new RealtimeOutgoingAudioSourceLibWebRTC(WTFMove(audioTrackPrivate)));
    }

private:
    explicit RealtimeOutgoingAudioSourceLibWebRTC(Ref<MediaStreamTrackPrivate>&&);
    ~RealtimeOutgoingAudioSourceLibWebRTC();

    // CheckedPtr interface
    uint32_t checkedPtrCount() const final { return CanMakeCheckedPtr::checkedPtrCount(); }
    uint32_t checkedPtrCountWithoutThreadCheck() const final { return CanMakeCheckedPtr::checkedPtrCountWithoutThreadCheck(); }
    void incrementCheckedPtrCount() const final { CanMakeCheckedPtr::incrementCheckedPtrCount(); }
    void decrementCheckedPtrCount() const final { CanMakeCheckedPtr::decrementCheckedPtrCount(); }

    void audioSamplesAvailable(const MediaTime&, const PlatformAudioData&, const AudioStreamDescription&, size_t) final;

    bool isReachingBufferedAudioDataHighLimit() final;
    bool isReachingBufferedAudioDataLowLimit() final;
    bool hasBufferedEnoughData() final;

    void pullAudioData();

    GUniquePtr<GstAudioConverter> m_sampleConverter;
    GstAudioInfo m_inputStreamDescription;
    GstAudioInfo m_outputStreamDescription;

    Lock m_adapterLock;
    GRefPtr<GstAdapter> m_adapter WTF_GUARDED_BY_LOCK(m_adapterLock);
    Vector<uint8_t> m_audioBuffer;
};

} // namespace WebCore

#endif // USE(LIBWEBRTC)
