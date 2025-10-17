/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 4, 2024.
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

#if USE(LIBWEBRTC) && PLATFORM(COCOA) && ENABLE(GPU_PROCESS)

#include "Connection.h"
#include "RemoteVideoFrameIdentifier.h"
#include "SharedPreferencesForWebProcess.h"
#include "SharedVideoFrame.h"
#include "VideoDecoderIdentifier.h"
#include "VideoEncoderIdentifier.h"
#include "WorkQueueMessageReceiver.h"
#include <WebCore/ProcessIdentity.h>
#include <WebCore/SharedMemory.h>
#include <WebCore/VideoCodecType.h>
#include <WebCore/VideoEncoderScalabilityMode.h>
#include <WebCore/WebRTCVideoDecoder.h>
#include <atomic>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadAssertions.h>

namespace IPC {
class Connection;
class Decoder;
class Semaphore;
}

namespace webrtc {
using LocalEncoder = void*;
}

namespace WebCore {
class FrameRateMonitor;
class PixelBufferConformerCV;
}

namespace WebKit {

class GPUConnectionToWebProcess;
class RemoteVideoFrameObjectHeap;
struct SharedVideoFrame;
class SharedVideoFrameReader;

class LibWebRTCCodecsProxy final : public IPC::WorkQueueMessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(LibWebRTCCodecsProxy);
public:
    static Ref<LibWebRTCCodecsProxy> create(GPUConnectionToWebProcess&, SharedPreferencesForWebProcess&);
    ~LibWebRTCCodecsProxy();

    void ref() const final { IPC::WorkQueueMessageReceiver::ref(); }
    void deref() const final { IPC::WorkQueueMessageReceiver::deref(); }

    void stopListeningForIPC(Ref<LibWebRTCCodecsProxy>&& refFromConnection);
    bool allowsExitUnderMemoryPressure() const;
    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const { return m_sharedPreferencesForWebProcess; }
    void updateSharedPreferencesForWebProcess(SharedPreferencesForWebProcess);
private:
    explicit LibWebRTCCodecsProxy(GPUConnectionToWebProcess&, SharedPreferencesForWebProcess&);
    void initialize();
    auto createDecoderCallback(VideoDecoderIdentifier, bool useRemoteFrames, bool enableAdditionalLogging);
    std::unique_ptr<WebCore::WebRTCVideoDecoder> createLocalDecoder(VideoDecoderIdentifier, WebCore::VideoCodecType, bool useRemoteFrames, bool enableAdditionalLogging);
    WorkQueue& workQueue() const { return m_queue; }
    Ref<WorkQueue> protectedWorkQueue() const { return m_queue; }

    Ref<IPC::Connection> protectedConnection() const { return m_connection; }

    // IPC::WorkQueueMessageReceiver overrides.
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    void createDecoder(VideoDecoderIdentifier, WebCore::VideoCodecType, const String& codecString, bool useRemoteFrames, bool enableAdditionalLogging, CompletionHandler<void(bool)>&&);
    void releaseDecoder(VideoDecoderIdentifier);
    void flushDecoder(VideoDecoderIdentifier, CompletionHandler<void()>&&);
    void setDecoderFormatDescription(VideoDecoderIdentifier, std::span<const uint8_t>, uint16_t width, uint16_t height);
    void decodeFrame(VideoDecoderIdentifier, int64_t timeStamp, std::span<const uint8_t>, CompletionHandler<void(bool)>&&);
    void setFrameSize(VideoDecoderIdentifier, uint16_t width, uint16_t height);

    void createEncoder(VideoEncoderIdentifier, WebCore::VideoCodecType, const String& codecString, const Vector<std::pair<String, String>>&, bool useLowLatency, bool useAnnexB, WebCore::VideoEncoderScalabilityMode, CompletionHandler<void(bool)>&&);
    void releaseEncoder(VideoEncoderIdentifier);
    void initializeEncoder(VideoEncoderIdentifier, uint16_t width, uint16_t height, unsigned startBitrate, unsigned maxBitrate, unsigned minBitrate, uint32_t maxFramerate);
    void encodeFrame(VideoEncoderIdentifier, SharedVideoFrame&&, int64_t timeStamp, std::optional<uint64_t> duration, bool shouldEncodeAsKeyFrame, CompletionHandler<void(bool)>&&);
    void flushEncoder(VideoEncoderIdentifier, CompletionHandler<void()>&&);
    void setEncodeRates(VideoEncoderIdentifier, uint32_t bitRate, uint32_t frameRate, CompletionHandler<void()>&&);
    void setSharedVideoFrameSemaphore(VideoEncoderIdentifier, IPC::Semaphore&&);
    void setSharedVideoFrameMemory(VideoEncoderIdentifier, WebCore::SharedMemory::Handle&&);
    void setRTCLoggingLevel(WTFLogLevel);

    void notifyEncoderResult(VideoEncoderIdentifier, bool);
    void notifyDecoderResult(VideoDecoderIdentifier, bool);

    struct Decoder {
        std::unique_ptr<WebCore::WebRTCVideoDecoder> webrtcDecoder;
        std::unique_ptr<WebCore::FrameRateMonitor> frameRateMonitor;
        Deque<CompletionHandler<void(bool)>> decodingCallbacks;
    };
    void doDecoderTask(VideoDecoderIdentifier, Function<void(Decoder&)>&&);

    struct Encoder {
        webrtc::LocalEncoder webrtcEncoder { nullptr };
        std::unique_ptr<SharedVideoFrameReader> frameReader;
        Deque<CompletionHandler<void(bool)>> encodingCallbacks;
    };
    Encoder* findEncoder(VideoEncoderIdentifier) WTF_REQUIRES_CAPABILITY(workQueue());

    Ref<IPC::Connection> m_connection;
    Ref<WorkQueue> m_queue;
    Ref<RemoteVideoFrameObjectHeap> m_videoFrameObjectHeap;
    WebCore::ProcessIdentity m_resourceOwner;
    SharedPreferencesForWebProcess m_sharedPreferencesForWebProcess;
    HashMap<VideoDecoderIdentifier, Decoder> m_decoders WTF_GUARDED_BY_CAPABILITY(workQueue());
    HashMap<VideoEncoderIdentifier, Encoder> m_encoders WTF_GUARDED_BY_CAPABILITY(workQueue());
    std::atomic<bool> m_hasEncodersOrDecoders { false };

    std::unique_ptr<WebCore::PixelBufferConformerCV> m_pixelBufferConformer;
};

}

#endif
