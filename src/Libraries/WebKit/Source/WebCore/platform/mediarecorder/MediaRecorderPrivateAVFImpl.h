/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 13, 2023.
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

#include "CAAudioStreamDescription.h"
#include "MediaRecorderPrivate.h"
#include "MediaRecorderPrivateEncoder.h"
#include <wtf/CheckedRef.h>
#include <wtf/TZoneMalloc.h>

using CVPixelBufferRef = struct __CVBuffer*;
typedef const struct opaqueCMFormatDescription* CMFormatDescriptionRef;

namespace WebCore {

class ContentType;
class Document;
class MediaStreamPrivate;
class WebAudioBufferList;

class MediaRecorderPrivateAVFImpl final
    : public MediaRecorderPrivate {
    WTF_MAKE_TZONE_ALLOCATED(MediaRecorderPrivateAVFImpl);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(MediaRecorderPrivateAVFImpl);
public:
    static std::unique_ptr<MediaRecorderPrivateAVFImpl> create(MediaStreamPrivate&, const MediaRecorderPrivateOptions&);
    ~MediaRecorderPrivateAVFImpl();

    static bool isTypeSupported(Document&, ContentType&);

private:
    explicit MediaRecorderPrivateAVFImpl(Ref<MediaRecorderPrivateEncoder>&&);

    // MediaRecorderPrivate
    void videoFrameAvailable(VideoFrame&, VideoFrameTimeMetadata) final;
    void fetchData(FetchDataCallback&&) final;
    void audioSamplesAvailable(const WTF::MediaTime&, const PlatformAudioData&, const AudioStreamDescription&, size_t) final;
    void startRecording(StartRecordingCallback&&) final;
    String mimeType() const final;

    void stopRecording(CompletionHandler<void()>&&) final;
    void pauseRecording(CompletionHandler<void()>&&) final;
    void resumeRecording(CompletionHandler<void()>&&) final;

    const Ref<MediaRecorderPrivateEncoder> m_encoder;
    RefPtr<VideoFrame> m_blackFrame;
    std::optional<CAAudioStreamDescription> m_description;
    std::unique_ptr<WebAudioBufferList> m_audioBuffer;
};

} // namespace WebCore

#endif // ENABLE(MEDIA_RECORDER)
