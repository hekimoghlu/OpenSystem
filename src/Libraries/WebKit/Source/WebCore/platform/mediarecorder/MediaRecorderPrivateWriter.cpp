/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 28, 2023.
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
#include "config.h"
#include "MediaRecorderPrivateWriter.h"

#if ENABLE(MEDIA_RECORDER)

#include "MediaRecorderPrivateWriterAVFObjC.h"
#include "MediaRecorderPrivateWriterWebM.h"
#include "MediaSample.h"
#include "MediaStrategy.h"
#include "PlatformStrategies.h"
#include <wtf/MediaTime.h>
#include <wtf/NativePromise.h>

namespace WebCore {

MediaRecorderPrivateWriter::MediaRecorderPrivateWriter() = default;
MediaRecorderPrivateWriter::~MediaRecorderPrivateWriter() = default;

std::unique_ptr<MediaRecorderPrivateWriter> MediaRecorderPrivateWriter::create(String type, MediaRecorderPrivateWriterListener& listener)
{
    auto containerType = [](const String& type) -> std::optional<MediaRecorderContainerType> {
        if (equalLettersIgnoringASCIICase(type, "video/mp4"_s) || equalLettersIgnoringASCIICase(type, "audio/mp4"_s))
            return MediaRecorderContainerType::Mp4;
#if ENABLE(MEDIA_RECORDER_WEBM)
        if (equalLettersIgnoringASCIICase(type, "video/webm"_s) || equalLettersIgnoringASCIICase(type, "audio/webm"_s))
            return MediaRecorderContainerType::WebM;
#endif
        return { };
    }(type);
    if (!containerType)
        return nullptr;

    return create(*containerType, listener);
}

std::unique_ptr<MediaRecorderPrivateWriter> MediaRecorderPrivateWriter::create(MediaRecorderContainerType type, MediaRecorderPrivateWriterListener& listener)
{
    if (hasPlatformStrategies()) {
        auto writer = platformStrategies()->mediaStrategy().createMediaRecorderPrivateWriter(type, listener);
        if (writer)
            return writer;
    }
    switch (type) {
    case MediaRecorderContainerType::Mp4:
        return MediaRecorderPrivateWriterAVFObjC::create(listener);
#if ENABLE(MEDIA_RECORDER_WEBM)
    case MediaRecorderContainerType::WebM:
        return MediaRecorderPrivateWriterWebM::create(listener);
#endif
    default:
        return nullptr;
    }
}

Ref<MediaRecorderPrivateWriter::WriterPromise> MediaRecorderPrivateWriter::writeFrames(Deque<UniqueRef<MediaSamplesBlock>>&& samples, const MediaTime& endTime)
{
    while (!samples.isEmpty())
        m_pendingFrames.append(samples.takeFirst());

    auto result = Result::Success;
    while (!m_pendingFrames.isEmpty() && result == Result::Success)
        result = writeFrame(m_pendingFrames.takeFirst().get());

    // End the segment if we succeded in writing all frames, otherwise we will retry them on the next call.
    if (m_pendingFrames.isEmpty())
        forceNewSegment(endTime);

    m_lastEndTime = endTime;

    return result == Result::Success ? WriterPromise::createAndResolve() : WriterPromise::createAndReject(result);
}

Ref<GenericPromise> MediaRecorderPrivateWriter::close()
{
    ASSERT(m_lastEndTime.isValid(), "writeFrames must have been called once");

    if (!m_pendingFrames.isEmpty())
        writeFrames({ }, m_lastEndTime); // Attempt one last time to write the frames we do have.

    m_pendingFrames.clear();
    return close(m_lastEndTime);
}

} // namespace WebCore

#endif // ENABLE(MEDIA_RECORDER)
