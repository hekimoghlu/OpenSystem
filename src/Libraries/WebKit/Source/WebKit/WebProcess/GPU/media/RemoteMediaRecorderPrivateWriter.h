/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 3, 2024.
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

#if ENABLE(GPU_PROCESS) && PLATFORM(COCOA) && ENABLE(MEDIA_RECORDER)

#include "RemoteMediaRecorderPrivateWriterIdentifier.h"
#include <WebCore/MediaRecorderPrivateWriter.h>
#include <WebCore/PlatformMediaResourceLoader.h>
#include <WebCore/PolicyChecker.h>
#include <WebCore/ResourceResponse.h>
#include <wtf/TZoneMalloc.h>

typedef const struct opaqueCMFormatDescription *CMFormatDescriptionRef;

namespace WebKit {

class GPUProcessConnection;

class RemoteMediaRecorderPrivateWriter final : public WebCore::MediaRecorderPrivateWriter {
    WTF_MAKE_TZONE_ALLOCATED(RemoteMediaRecorderPrivateWriter);
public:
    static std::unique_ptr<MediaRecorderPrivateWriter> create(GPUProcessConnection&, WebCore::MediaRecorderPrivateWriterListener&);

private:
    RemoteMediaRecorderPrivateWriter(GPUProcessConnection&, WebCore::MediaRecorderPrivateWriterListener&);
    std::optional<uint8_t> addAudioTrack(const WebCore::AudioInfo&) final;
    std::optional<uint8_t> addVideoTrack(const WebCore::VideoInfo&, const std::optional<CGAffineTransform>&) final;
    bool allTracksAdded() final;
    Ref<WriterPromise> writeFrames(Deque<UniqueRef<WebCore::MediaSamplesBlock>>&&, const MediaTime&) final;
    Result writeFrame(const WebCore::MediaSamplesBlock&) final;
    void forceNewSegment(const MediaTime&) final { };
    Ref<GenericPromise> close() final;
    Ref<GenericPromise> close(const MediaTime&) final;

    bool m_isClosed { false };
    ThreadSafeWeakPtr<GPUProcessConnection> m_gpuProcessConnection;
    ThreadSafeWeakPtr<WebCore::MediaRecorderPrivateWriterListener> m_listener;
    const RemoteMediaRecorderPrivateWriterIdentifier m_remoteMediaRecorderPrivateWriterIdentifier;
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && PLATFORM(COCOA) ENABLE(MEDIA_RECORDER)

