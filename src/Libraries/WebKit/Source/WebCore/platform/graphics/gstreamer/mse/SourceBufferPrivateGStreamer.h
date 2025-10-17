/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 15, 2022.
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

#include "ContentType.h"
#include "MediaPlayerPrivateGStreamerMSE.h"
#include "MediaSourceTrackGStreamer.h"
#include "SourceBufferPrivate.h"
#include "SourceBufferPrivateClient.h"
#include "TrackPrivateBaseGStreamer.h"
#include "WebKitMediaSourceGStreamer.h"
#include <optional>
#include <wtf/LoggerHelper.h>
#include <wtf/StdUnorderedMap.h>

namespace WebCore {

using TrackID = uint64_t;

class AppendPipeline;
class MediaSourcePrivateGStreamer;

class SourceBufferPrivateGStreamer final : public SourceBufferPrivate, public CanMakeWeakPtr<SourceBufferPrivateGStreamer> {
public:
    static bool isContentTypeSupported(const ContentType&);
    static Ref<SourceBufferPrivateGStreamer> create(MediaSourcePrivateGStreamer&, const ContentType&);
    ~SourceBufferPrivateGStreamer();

    constexpr MediaPlatformType platformType() const final { return MediaPlatformType::GStreamer; }

    Ref<MediaPromise> appendInternal(Ref<SharedBuffer>&&) final;
    void resetParserStateInternal() final;
    void removedFromMediaSource() final;

    void flush(TrackID) final;
    void enqueueSample(Ref<MediaSample>&&, TrackID) final;
    void allSamplesInTrackEnqueued(TrackID) final;
    bool isReadyForMoreSamples(TrackID) final;

    bool precheckInitializationSegment(const InitializationSegment&) final;
    void processInitializationSegment(std::optional<InitializationSegment>&&) final;

    void didReceiveAllPendingSamples();
    void appendParsingFailed();

    auto& tracks() const { return m_tracks; }

    ContentType type() const { return m_type; }

    std::optional<TrackID> tryRegisterTrackId(TrackID);
    bool tryUnregisterTrackId(TrackID);

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const final { return m_logger.get(); }
    ASCIILiteral logClassName() const override { return "SourceBufferPrivateGStreamer"_s; }
    uint64_t logIdentifier() const final { return m_logIdentifier; }
    WTFLogChannel& logChannel() const final;
    const Logger& sourceBufferLogger() const final { return m_logger; }
    uint64_t sourceBufferLogIdentifier() final { return logIdentifier(); }
#endif

    size_t platformMaximumBufferSize() const override;
    size_t platformEvictionThreshold() const final;

private:
    friend class AppendPipeline;

    SourceBufferPrivateGStreamer(MediaSourcePrivateGStreamer&, const ContentType&);
    RefPtr<MediaPlayerPrivateGStreamerMSE> player() const;

    void notifyClientWhenReadyForMoreSamples(TrackID) override;

    void detach() final;

    bool m_hasBeenRemovedFromMediaSource { false };
    ContentType m_type;
    std::unique_ptr<AppendPipeline> m_appendPipeline;
    StdUnorderedMap<TrackID, RefPtr<MediaSourceTrackGStreamer>> m_tracks;
    std::optional<MediaPromise::Producer> m_appendPromise;

#if !RELEASE_LOG_DISABLED
    Ref<const Logger> m_logger;
    const uint64_t m_logIdentifier;
#endif
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::SourceBufferPrivateGStreamer)
static bool isType(const WebCore::SourceBufferPrivate& sourceBuffer) { return sourceBuffer.platformType() == WebCore::MediaPlatformType::GStreamer; }
SPECIALIZE_TYPE_TRAITS_END()

#endif
