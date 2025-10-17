/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 5, 2022.
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

#include <WebCore/MediaStrategy.h>
#include <atomic>

namespace WebKit {

class WebMediaStrategy final : public WebCore::MediaStrategy {
public:
    virtual ~WebMediaStrategy();

#if ENABLE(GPU_PROCESS)
    void setUseGPUProcess(bool useGPUProcess) { m_useGPUProcess = useGPUProcess; }
#endif

private:
#if ENABLE(WEB_AUDIO)
    Ref<WebCore::AudioDestination> createAudioDestination(WebCore::AudioIOCallback&,
        const String& inputDeviceId, unsigned numberOfInputChannels, unsigned numberOfOutputChannels, float sampleRate) override;
#endif
    std::unique_ptr<WebCore::NowPlayingManager> createNowPlayingManager() const final;
    bool hasThreadSafeMediaSourceSupport() const final;
#if ENABLE(MEDIA_SOURCE)
    void enableMockMediaSource() final;
#endif
#if PLATFORM(COCOA) && ENABLE(MEDIA_RECORDER)
    std::unique_ptr<WebCore::MediaRecorderPrivateWriter> createMediaRecorderPrivateWriter(MediaRecorderContainerType, WebCore::MediaRecorderPrivateWriterListener&) const final;
#endif

#if ENABLE(GPU_PROCESS)
    std::atomic<bool> m_useGPUProcess { false };
#endif
};

} // namespace WebKit
