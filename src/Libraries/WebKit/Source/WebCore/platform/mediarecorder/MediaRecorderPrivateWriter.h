/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 8, 2022.
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

#if ENABLE(MEDIA_RECORDER)

#include <memory>
#include <optional>
#include <wtf/Deque.h>
#include <wtf/Forward.h>
#include <wtf/MediaTime.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadSafeWeakPtr.h>

typedef const struct opaqueCMFormatDescription* CMFormatDescriptionRef;
struct CGAffineTransform;

namespace WebCore {

class MediaSamplesBlock;
struct AudioInfo;
struct VideoInfo;

class MediaRecorderPrivateWriterListener : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<MediaRecorderPrivateWriterListener> {
public:
    virtual void appendData(std::span<const uint8_t>) = 0;
    virtual ~MediaRecorderPrivateWriterListener() = default;
};

enum class MediaRecorderContainerType : uint8_t {
    Mp4,
    WebM
};

class MediaRecorderPrivateWriter {
    WTF_MAKE_TZONE_ALLOCATED(MediaRecorderPrivateWriter);
public:
    WEBCORE_EXPORT static std::unique_ptr<MediaRecorderPrivateWriter> create(String type, MediaRecorderPrivateWriterListener&);
    WEBCORE_EXPORT static std::unique_ptr<MediaRecorderPrivateWriter> create(MediaRecorderContainerType, MediaRecorderPrivateWriterListener&);

    WEBCORE_EXPORT MediaRecorderPrivateWriter();
    WEBCORE_EXPORT virtual ~MediaRecorderPrivateWriter();

    virtual std::optional<uint8_t> addAudioTrack(const AudioInfo&) = 0;
    virtual std::optional<uint8_t> addVideoTrack(const VideoInfo&, const std::optional<CGAffineTransform>&) = 0;
    virtual bool allTracksAdded() = 0;
    enum class Result : uint8_t { Success, Failure, NotReady };
    using WriterPromise = NativePromise<void, Result>;
    WEBCORE_EXPORT virtual Ref<WriterPromise> writeFrames(Deque<UniqueRef<MediaSamplesBlock>>&&, const MediaTime&);
    WEBCORE_EXPORT virtual Ref<GenericPromise> close();

private:
    virtual Result writeFrame(const MediaSamplesBlock&) = 0;
    virtual void forceNewSegment(const MediaTime&) = 0;
    virtual Ref<GenericPromise> close(const MediaTime&) = 0;
    Deque<UniqueRef<MediaSamplesBlock>> m_pendingFrames;
    MediaTime m_lastEndTime { MediaTime::invalidTime() };
};

} // namespace WebCore

#endif // ENABLE(MEDIA_RECORDER)
