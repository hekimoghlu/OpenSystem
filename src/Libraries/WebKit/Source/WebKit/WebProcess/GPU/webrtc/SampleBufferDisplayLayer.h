/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 7, 2023.
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

#if PLATFORM(COCOA) && ENABLE(GPU_PROCESS) && ENABLE(MEDIA_STREAM)

#include "GPUProcessConnection.h"
#include "MessageReceiver.h"
#include "SampleBufferDisplayLayerIdentifier.h"
#include "SharedVideoFrame.h"
#include <WebCore/SampleBufferDisplayLayer.h>
#include <wtf/Identified.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
enum class VideoFrameRotation : uint16_t;
}

namespace WebKit {

class SampleBufferDisplayLayerManager;

class SampleBufferDisplayLayer final : public WebCore::SampleBufferDisplayLayer, public IPC::MessageReceiver, public GPUProcessConnection::Client, public Identified<SampleBufferDisplayLayerIdentifier> {
public:
    static Ref<SampleBufferDisplayLayer> create(SampleBufferDisplayLayerManager&, WebCore::SampleBufferDisplayLayerClient&);
    ~SampleBufferDisplayLayer();

    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    WebCore::LayerHostingContextID hostingContextID() const final { return m_hostingContextID.value_or(0); }

    WTF_ABSTRACT_THREAD_SAFE_REF_COUNTED_AND_CAN_MAKE_WEAK_PTR_IMPL;

private:
    SampleBufferDisplayLayer(SampleBufferDisplayLayerManager&, WebCore::SampleBufferDisplayLayerClient&);

    // WebCore::SampleBufferDisplayLayer
    void initialize(bool hideRootLayer, WebCore::IntSize, bool shouldMaintainAspectRatio, CompletionHandler<void(bool)>&&) final;
#if !RELEASE_LOG_DISABLED
    void setLogIdentifier(String&&) final;
#endif
    bool didFail() const final;
    void updateDisplayMode(bool hideDisplayLayer, bool hideRootLayer) final;
    void updateBoundsAndPosition(CGRect, std::optional<WTF::MachSendRight>&&) final;
    void flush() final;
    void flushAndRemoveImage() final;
    void play() final;
    void pause() final;
    void enqueueBlackFrameFrom(const WebCore::VideoFrame&) final;
    void enqueueVideoFrame(WebCore::VideoFrame&) final;
    void clearVideoFrames() final;
    PlatformLayer* rootLayer() final;
    void setShouldMaintainAspectRatio(bool) final;

    // GPUProcessConnection::Client
    void gpuProcessConnectionDidClose(GPUProcessConnection&) final;

    void setDidFail(bool);

    ThreadSafeWeakPtr<GPUProcessConnection> m_gpuProcessConnection;
    WeakPtr<SampleBufferDisplayLayerManager> m_manager;
    Ref<IPC::Connection> m_connection;

    PlatformLayerContainer m_videoLayer;
    bool m_didFail { false };
    bool m_paused { false };

    SharedVideoFrameWriter m_sharedVideoFrameWriter;
    std::optional<WebCore::LayerHostingContextID> m_hostingContextID;
};

}

#endif // PLATFORM(COCOA) && ENABLE(GPU_PROCESS) && ENABLE(MEDIA_STREAM)
