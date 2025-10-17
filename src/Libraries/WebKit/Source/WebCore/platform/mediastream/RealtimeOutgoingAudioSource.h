/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 3, 2025.
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

#include "LibWebRTCMacros.h"
#include "MediaStreamTrackPrivate.h"
#include "Timer.h"

ALLOW_UNUSED_PARAMETERS_BEGIN
ALLOW_COMMA_BEGIN

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN
#include <webrtc/api/media_stream_interface.h>
WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

ALLOW_COMMA_END
ALLOW_UNUSED_PARAMETERS_END

#include <wtf/Lock.h>
#include <wtf/LoggerHelper.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace webrtc {
class AudioTrackInterface;
class AudioTrackSinkInterface;
}

namespace WebCore {

class RealtimeOutgoingAudioSource
    : public ThreadSafeRefCounted<RealtimeOutgoingAudioSource, WTF::DestructionThread::Main>
    , public webrtc::AudioSourceInterface
    , private MediaStreamTrackPrivateObserver
    , private RealtimeMediaSource::AudioSampleObserver
#if !RELEASE_LOG_DISABLED
    , private LoggerHelper
#endif
{
public:
    static Ref<RealtimeOutgoingAudioSource> create(Ref<MediaStreamTrackPrivate>&& audioSource);

    ~RealtimeOutgoingAudioSource();

    void start() { observeSource(); }
    void stop() { unobserveSource(); }

    void setSource(Ref<MediaStreamTrackPrivate>&&);
    MediaStreamTrackPrivate& source() const { return m_audioSource.get(); }

protected:
    explicit RealtimeOutgoingAudioSource(Ref<MediaStreamTrackPrivate>&&);

    bool isSilenced() const { return m_muted || !m_enabled; }

    void sendAudioFrames(const void* audioData, int bitsPerSample, int sampleRate, size_t numberOfChannels, size_t numberOfFrames);

#if !RELEASE_LOG_DISABLED
    // LoggerHelper API
    const Logger& logger() const final { return m_audioSource->logger(); }
    uint64_t logIdentifier() const final { return m_audioSource->logIdentifier(); }
    ASCIILiteral logClassName() const final { return "RealtimeOutgoingAudioSource"_s; }
    WTFLogChannel& logChannel() const final;
#endif

private:
    // webrtc::AudioSourceInterface API
    void AddSink(webrtc::AudioTrackSinkInterface*) final;
    void RemoveSink(webrtc::AudioTrackSinkInterface*) final;

    void AddRef() const final { ref(); }
    webrtc::RefCountReleaseStatus Release() const final
    {
        auto result = refCount() - 1;
        deref();
        return result ? webrtc::RefCountReleaseStatus::kOtherRefsRemained : webrtc::RefCountReleaseStatus::kDroppedLastRef;
    }

    SourceState state() const final { return kLive; }
    bool remote() const final { return false; }
    void RegisterObserver(webrtc::ObserverInterface*) final { }
    void UnregisterObserver(webrtc::ObserverInterface*) final { }

    void observeSource();
    void unobserveSource();

    void sourceMutedChanged();
    void sourceEnabledChanged();

    virtual bool isReachingBufferedAudioDataHighLimit() { return false; };
    virtual bool isReachingBufferedAudioDataLowLimit() { return false; };
    virtual bool hasBufferedEnoughData() { return false; };
    virtual void sourceUpdated() { }

    // MediaStreamTrackPrivateObserver API
    void trackMutedChanged(MediaStreamTrackPrivate&) final { sourceMutedChanged(); }
    void trackEnabledChanged(MediaStreamTrackPrivate&) final { sourceEnabledChanged(); }
    void trackEnded(MediaStreamTrackPrivate&) final { }
    void trackSettingsChanged(MediaStreamTrackPrivate&) final { }

    void initializeConverter();

    Ref<MediaStreamTrackPrivate> m_audioSource;
    bool m_muted { false };
    bool m_enabled { true };

    mutable Lock m_sinksLock;
    UncheckedKeyHashSet<webrtc::AudioTrackSinkInterface*> m_sinks WTF_GUARDED_BY_LOCK(m_sinksLock);

#if !RELEASE_LOG_DISABLED
    size_t m_chunksSent { 0 };
#endif
};

} // namespace WebCore

#endif // USE(LIBWEBRTC)
