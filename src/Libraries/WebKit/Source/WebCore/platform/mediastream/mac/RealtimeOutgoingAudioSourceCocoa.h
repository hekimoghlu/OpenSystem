/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 5, 2022.
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

#include "AudioSampleDataSource.h"
#include "RealtimeOutgoingAudioSource.h"
#include <wtf/CheckedRef.h>
#include <wtf/TZoneMalloc.h>

namespace webrtc {
class AudioTrackInterface;
class AudioTrackSinkInterface;
}

namespace WebCore {

class RealtimeOutgoingAudioSourceCocoa final : public RealtimeOutgoingAudioSource, public CanMakeCheckedPtr<RealtimeOutgoingAudioSourceCocoa> {
    WTF_MAKE_TZONE_ALLOCATED(RealtimeOutgoingAudioSourceCocoa);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RealtimeOutgoingAudioSourceCocoa);
public:
    static Ref<RealtimeOutgoingAudioSourceCocoa> create(Ref<MediaStreamTrackPrivate>&& audioSource) { return adoptRef(*new RealtimeOutgoingAudioSourceCocoa(WTFMove(audioSource))); }

private:
    explicit RealtimeOutgoingAudioSourceCocoa(Ref<MediaStreamTrackPrivate>&&);
    ~RealtimeOutgoingAudioSourceCocoa();

    // CheckedPtr interface
    uint32_t checkedPtrCount() const final { return CanMakeCheckedPtr::checkedPtrCount(); }
    uint32_t checkedPtrCountWithoutThreadCheck() const final { return CanMakeCheckedPtr::checkedPtrCountWithoutThreadCheck(); }
    void incrementCheckedPtrCount() const final { CanMakeCheckedPtr::incrementCheckedPtrCount(); }
    void decrementCheckedPtrCount() const final { CanMakeCheckedPtr::decrementCheckedPtrCount(); }

    void audioSamplesAvailable(const MediaTime&, const PlatformAudioData&, const AudioStreamDescription&, size_t) final;

    bool isReachingBufferedAudioDataHighLimit() final;
    bool isReachingBufferedAudioDataLowLimit() final;
    bool hasBufferedEnoughData() final;
    void sourceUpdated() final;

    void pullAudioData();

    Ref<AudioSampleDataSource> m_sampleConverter;
    std::optional<CAAudioStreamDescription> m_inputStreamDescription;
    std::optional<CAAudioStreamDescription> m_outputStreamDescription;

    Vector<uint8_t> m_audioBuffer;
    uint64_t m_readCount { 0 };
    uint64_t m_writeCount { 0 };
    bool m_skippingAudioData { false };
};

} // namespace WebCore

#endif // USE(LIBWEBRTC)
