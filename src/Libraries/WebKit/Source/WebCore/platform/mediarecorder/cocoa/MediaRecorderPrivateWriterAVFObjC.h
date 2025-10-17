/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 16, 2025.
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

#include "MediaRecorderPrivateWriter.h"
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>

OBJC_CLASS AVAssetWriter;
OBJC_CLASS AVAssetWriterInput;
OBJC_CLASS WebAVAssetWriterDelegate;
typedef const struct opaqueCMFormatDescription *CMFormatDescriptionRef;

namespace WebCore {

class MediaRecorderPrivateWriterAVFObjC : public MediaRecorderPrivateWriter {
    WTF_MAKE_TZONE_ALLOCATED(MediaRecorderPrivateWriterAVFObjC);
public:
    static std::unique_ptr<MediaRecorderPrivateWriter> create(MediaRecorderPrivateWriterListener&);
    ~MediaRecorderPrivateWriterAVFObjC();

private:
    MediaRecorderPrivateWriterAVFObjC(RetainPtr<AVAssetWriter>&&, MediaRecorderPrivateWriterListener&);

    std::optional<uint8_t> addAudioTrack(const AudioInfo&) final;
    std::optional<uint8_t> addVideoTrack(const VideoInfo&, const std::optional<CGAffineTransform>&) final;
    bool allTracksAdded() final;
    Result writeFrame(const MediaSamplesBlock&) final;
    void forceNewSegment(const WTF::MediaTime&) final;
    Ref<GenericPromise> close(const WTF::MediaTime&) final;

    RetainPtr<AVAssetWriterInput> m_audioAssetWriterInput;
    RetainPtr<AVAssetWriterInput> m_videoAssetWriterInput;
    bool m_hasAddedVideoFrame { false };

    uint8_t m_currentTrackIndex { 0 };
    uint8_t m_audioTrackIndex { 0 };
    uint8_t m_videoTrackIndex  { 0 };
    RetainPtr<CMFormatDescriptionRef> m_audioDescription;
    RetainPtr<CMFormatDescriptionRef> m_videoDescription;
    const RetainPtr<WebAVAssetWriterDelegate> m_delegate;
    const RetainPtr<AVAssetWriter> m_writer;
};

} // namespace WebCore

#endif // ENABLE(MEDIA_RECORDER)
