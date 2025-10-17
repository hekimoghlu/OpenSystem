/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 6, 2024.
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

#if ENABLE(GPU_PROCESS) && ENABLE(WEB_AUDIO)

#include "Connection.h"
#include "GPUProcessConnection.h"
#include "IPCSemaphore.h"
#include "RemoteAudioDestinationIdentifier.h"
#include <WebCore/AudioDestinationResampler.h>
#include <WebCore/AudioIOCallback.h>
#include <wtf/CrossThreadQueue.h>
#include <wtf/MediaTime.h>
#include <wtf/Threading.h>

#if PLATFORM(COCOA)
#include "SharedCARingBuffer.h"
#endif

#if PLATFORM(COCOA)
namespace WebCore {
class WebAudioBufferList;
}
#endif

namespace WebKit {

class RemoteAudioDestinationProxy : public WebCore::AudioDestinationResampler, public GPUProcessConnection::Client, public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<RemoteAudioDestinationProxy> {
    WTF_MAKE_NONCOPYABLE(RemoteAudioDestinationProxy);
public:
    using AudioIOCallback = WebCore::AudioIOCallback;

    static Ref<RemoteAudioDestinationProxy> create(AudioIOCallback&, const String& inputDeviceId, unsigned numberOfInputChannels, unsigned numberOfOutputChannels, float sampleRate);

    RemoteAudioDestinationProxy(AudioIOCallback&, const String& inputDeviceId, unsigned numberOfInputChannels, unsigned numberOfOutputChannels, float sampleRate);
    ~RemoteAudioDestinationProxy();

    WTF_ABSTRACT_THREAD_SAFE_REF_COUNTED_AND_CAN_MAKE_WEAK_PTR_IMPL;

private:
    void startRendering(CompletionHandler<void(bool)>&&) override;
    void stopRendering(CompletionHandler<void(bool)>&&) override;
    MediaTime outputLatency() const final;

    void startRenderingThread();
    void stopRenderingThread();
    void renderAudio(unsigned frameCount);

    IPC::Connection* connection();
    IPC::Connection* existingConnection();

    // GPUProcessConnection::Client.
    void gpuProcessConnectionDidClose(GPUProcessConnection&) final;

    uint32_t totalFrameCount() const;

    Markable<RemoteAudioDestinationIdentifier> m_destinationID; // Call destinationID() getter to make sure the destinationID is valid.

    static uint8_t s_realtimeThreadCount;
    static constexpr uint8_t s_maximumConcurrentRealtimeThreads { 3 };

    ThreadSafeWeakPtr<GPUProcessConnection> m_gpuProcessConnection;
#if PLATFORM(COCOA)
    std::unique_ptr<ProducerSharedCARingBuffer> m_ringBuffer;
    std::unique_ptr<WebCore::WebAudioBufferList> m_audioBufferList;
    uint64_t m_currentFrame { 0 };
#endif
    IPC::Semaphore m_renderSemaphore;

    String m_inputDeviceId;
    unsigned m_numberOfInputChannels;
    float m_remoteSampleRate;
    size_t m_audioUnitLatency;

    RefPtr<Thread> m_renderThread;
    RefPtr<WebCore::SharedMemory> m_frameCount;
    uint32_t m_lastFrameCount { 0 };
    std::atomic<bool> m_shouldStopThread { false };
    bool m_isRealtimeThread { false };
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS) && ENABLE(WEB_AUDIO)
