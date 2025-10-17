/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 27, 2022.
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

#include "AudioTrackPrivate.h"
#include "InbandTextTrackPrivate.h"
#include "MediaDescription.h"
#include "PlatformMediaError.h"
#include "VideoTrackPrivate.h"
#include <wtf/MediaTime.h>
#include <wtf/Ref.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

class MediaSample;
class MediaDescription;
class PlatformTimeRanges;

struct SourceBufferEvictionData {
    uint64_t contentSize { 0 };
    int64_t evictableSize { 0 };
    uint64_t maximumBufferSize { 0 };
    size_t numMediaSamples { 0 };

    bool operator!=(const SourceBufferEvictionData& other)
    {
        return contentSize != other.contentSize || evictableSize != other.evictableSize || maximumBufferSize != other.maximumBufferSize || numMediaSamples != other.numMediaSamples;
    }

    void clear()
    {
        contentSize = 0;
        evictableSize = 0;
        numMediaSamples = 0;
    }
};

class SourceBufferPrivateClient : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<SourceBufferPrivateClient> {
public:
    virtual ~SourceBufferPrivateClient() = default;

    struct InitializationSegment {
        MediaTime duration;

        struct AudioTrackInformation {
            RefPtr<MediaDescription> description;
            RefPtr<AudioTrackPrivate> track;

            RefPtr<MediaDescription> protectedDescription() const { return description; }
            RefPtr<AudioTrackPrivate> protectedTrack() const { return track; }
        };
        Vector<AudioTrackInformation> audioTracks;

        struct VideoTrackInformation {
            RefPtr<MediaDescription> description;
            RefPtr<VideoTrackPrivate> track;

            RefPtr<MediaDescription> protectedDescription() const { return description; }
            RefPtr<VideoTrackPrivate> protectedTrack() const { return track; }
        };
        Vector<VideoTrackInformation> videoTracks;

        struct TextTrackInformation {
            RefPtr<MediaDescription> description;
            RefPtr<InbandTextTrackPrivate> track;

            RefPtr<MediaDescription> protectedDescription() const { return description; }
            RefPtr<InbandTextTrackPrivate> protectedTrack() const { return track; }
        };
        Vector<TextTrackInformation> textTracks;
    };

    virtual Ref<MediaPromise> sourceBufferPrivateDidReceiveInitializationSegment(InitializationSegment&&) = 0;
    virtual Ref<MediaPromise> sourceBufferPrivateBufferedChanged(const Vector<PlatformTimeRanges>&) = 0;
    virtual Ref<MediaPromise> sourceBufferPrivateDurationChanged(const MediaTime&) = 0;
    virtual void sourceBufferPrivateHighestPresentationTimestampChanged(const MediaTime&) = 0;
    virtual void sourceBufferPrivateDidDropSample() = 0;
    virtual void sourceBufferPrivateDidReceiveRenderingError(int64_t errorCode) = 0;
    virtual void sourceBufferPrivateEvictionDataChanged(const SourceBufferEvictionData&) { }
    virtual Ref<MediaPromise> sourceBufferPrivateDidAttach(InitializationSegment&&) = 0;
};

} // namespace WebCore

namespace WTF {
template<>
struct LogArgument<WebCore::SourceBufferEvictionData> {
    static String toString(const WebCore::SourceBufferEvictionData& evictionData)
    {
        return makeString("{ contentSize:"_s, evictionData.contentSize, " evictableData:"_s, evictionData.evictableSize, " maximumBufferSize:"_s, evictionData.maximumBufferSize, " numSamples:"_s, evictionData.numMediaSamples, " }"_s);
    }
};

} // namespace WTF

#endif // ENABLE(MEDIA_SOURCE)
