/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 9, 2025.
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

#if ENABLE(MEDIA_RECORDER_WEBM)

#include "MediaRecorderPrivateWriter.h"
#include <wtf/Forward.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class MediaRecorderPrivateWriterWebMDelegate;

class MediaRecorderPrivateWriterWebM final : public MediaRecorderPrivateWriter {
    WTF_MAKE_TZONE_ALLOCATED(MediaRecorderPrivateWriterWebM);
public:
    static std::unique_ptr<MediaRecorderPrivateWriter> create(MediaRecorderPrivateWriterListener&);

    ~MediaRecorderPrivateWriterWebM();
private:
    MediaRecorderPrivateWriterWebM(MediaRecorderPrivateWriterListener&);

    std::optional<uint8_t> addAudioTrack(const AudioInfo&) final;
    std::optional<uint8_t> addVideoTrack(const VideoInfo&, const std::optional<CGAffineTransform>&) final;
    bool allTracksAdded() final { return true; }
    Result writeFrame(const MediaSamplesBlock&) final;
    void forceNewSegment(const WTF::MediaTime&) final;
    Ref<GenericPromise> close(const WTF::MediaTime&) final;

    const UniqueRef<MediaRecorderPrivateWriterWebMDelegate> m_delegate;
};

} // namespace WebCore

#endif // ENABLE(MEDIA_RECORDER)
