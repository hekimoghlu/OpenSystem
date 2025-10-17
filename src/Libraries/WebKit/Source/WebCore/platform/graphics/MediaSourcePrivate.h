/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 23, 2025.
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

#if ENABLE(MEDIA_SOURCE)

#include "MediaPlayer.h"
#include "PlatformTimeRanges.h"
#include <wtf/Forward.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/Vector.h>

namespace WebCore {

class ContentType;
class MediaPlayerPrivateInterface;
class SourceBufferPrivate;
#if ENABLE(LEGACY_ENCRYPTED_MEDIA)
class LegacyCDMSession;
#endif
enum class MediaSourceReadyState;

enum class MediaSourcePrivateAddStatus : uint8_t {
    Ok,
    NotSupported,
    ReachedIdLimit
};

enum class MediaSourcePrivateEndOfStreamStatus : uint8_t {
    NoError,
    NetworkError,
    DecodeError
};

class WEBCORE_EXPORT MediaSourcePrivate
    : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<MediaSourcePrivate> {
public:
    typedef Vector<String> CodecsArray;

    using AddStatus = MediaSourcePrivateAddStatus;
    using EndOfStreamStatus = MediaSourcePrivateEndOfStreamStatus;

    explicit MediaSourcePrivate(MediaSourcePrivateClient&);
    virtual ~MediaSourcePrivate();

    RefPtr<MediaSourcePrivateClient> client() const;
    virtual RefPtr<MediaPlayerPrivateInterface> player() const = 0;
    virtual void setPlayer(MediaPlayerPrivateInterface*) = 0;
    virtual void shutdown();

    virtual constexpr MediaPlatformType platformType() const = 0;
    virtual AddStatus addSourceBuffer(const ContentType&, RefPtr<SourceBufferPrivate>&) = 0;
    virtual void removeSourceBuffer(SourceBufferPrivate&);
    void sourceBufferPrivateDidChangeActiveState(SourceBufferPrivate&, bool active);
    virtual void notifyActiveSourceBuffersChanged() = 0;
    virtual void durationChanged(const MediaTime&); // Base class method must be called in overrides. Must be thread-safe
    virtual void bufferedChanged(const PlatformTimeRanges&); // Base class method must be called in overrides. Must be thread-safe.
    virtual void trackBufferedChanged(SourceBufferPrivate&, Vector<PlatformTimeRanges>&&);

    virtual MediaPlayer::ReadyState mediaPlayerReadyState() const = 0;
    virtual void setMediaPlayerReadyState(MediaPlayer::ReadyState) = 0;
    virtual void markEndOfStream(EndOfStreamStatus) { m_isEnded = true; }
    virtual void unmarkEndOfStream() { m_isEnded = false; }
    bool isEnded() const { return m_isEnded; }

    virtual MediaSourceReadyState readyState() const { return m_readyState; }
    virtual void setReadyState(MediaSourceReadyState readyState) { m_readyState = readyState; }
    void setLiveSeekableRange(const PlatformTimeRanges&);
    const PlatformTimeRanges& liveSeekableRange() const;
    void clearLiveSeekableRange();

    MediaTime currentTime() const;

    Ref<MediaTimePromise> waitForTarget(const SeekTarget&);
    Ref<MediaPromise> seekToTime(const MediaTime&);

    virtual void setTimeFudgeFactor(const MediaTime& fudgeFactor) { m_timeFudgeFactor = fudgeFactor; }
    MediaTime timeFudgeFactor() const { return m_timeFudgeFactor; }

    MediaTime duration() const;
    PlatformTimeRanges buffered() const;
    PlatformTimeRanges seekable() const;

    bool hasBufferedData() const;
    bool hasFutureTime(const MediaTime& currentTime) const;
    bool timeIsProgressing() const;
    static constexpr MediaTime futureDataThreshold() { return MediaTime { 1001, 24000 }; }
    bool hasFutureTime(const MediaTime& currentTime, const MediaTime& threshold) const;
    bool hasAudio() const;
    bool hasVideo() const;

    void setStreaming(bool value) { m_streaming = value; }
    bool streaming() const { return m_streaming; }
    void setStreamingAllowed(bool value) { m_streamingAllowed = value; }
    bool streamingAllowed() const { return m_streamingAllowed; }

#if ENABLE(LEGACY_ENCRYPTED_MEDIA)
    void setCDMSession(LegacyCDMSession*);
#endif

protected:
    MediaSourcePrivate(MediaSourcePrivateClient&, GuaranteedSerialFunctionDispatcher&);
    void ensureOnDispatcher(Function<void()>&&) const;

    Vector<RefPtr<SourceBufferPrivate>> m_sourceBuffers;
    Vector<SourceBufferPrivate*> m_activeSourceBuffers;
    std::atomic<bool> m_isEnded { false }; // Set on MediaSource's dispatcher.
    std::atomic<MediaSourceReadyState> m_readyState; // Set on MediaSource's dispatcher.
    const Ref<GuaranteedSerialFunctionDispatcher> m_dispatcher; // SerialFunctionDispatcher the SourceBufferPrivate/MediaSourcePrivate is running on.

private:
    mutable Lock m_lock;
    MediaTime m_duration WTF_GUARDED_BY_LOCK(m_lock) { MediaTime::invalidTime() };
    PlatformTimeRanges m_buffered WTF_GUARDED_BY_LOCK(m_lock);
    PlatformTimeRanges m_liveSeekable WTF_GUARDED_BY_LOCK(m_lock);
    std::atomic<bool> m_streaming { false };
    std::atomic<bool> m_streamingAllowed { false };
    MediaTime m_timeFudgeFactor;
    const ThreadSafeWeakPtr<MediaSourcePrivateClient> m_client;
};

String convertEnumerationToString(MediaSourcePrivate::AddStatus);
String convertEnumerationToString(MediaSourcePrivate::EndOfStreamStatus);

} // namespace WebCore

namespace WTF {

template<typename Type> struct LogArgument;

template <>
struct LogArgument<WebCore::MediaSourcePrivate::AddStatus> {
    static String toString(const WebCore::MediaSourcePrivate::AddStatus status)
    {
        return convertEnumerationToString(status);
    }
};

template <>
struct LogArgument<WebCore::MediaSourcePrivate::EndOfStreamStatus> {
    static String toString(const WebCore::MediaSourcePrivate::EndOfStreamStatus status)
    {
        return convertEnumerationToString(status);
    }
};

} // namespace WTF

#endif
