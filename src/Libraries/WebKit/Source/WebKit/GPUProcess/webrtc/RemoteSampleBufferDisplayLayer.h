/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 13, 2024.
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

#include "LayerHostingContext.h"
#include "MessageReceiver.h"
#include "MessageSender.h"
#include "RemoteSampleBufferDisplayLayerManager.h"
#include "RemoteVideoFrameIdentifier.h"
#include "SampleBufferDisplayLayerIdentifier.h"
#include "SharedVideoFrame.h"
#include <WebCore/SampleBufferDisplayLayer.h>
#include <wtf/MediaTime.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/ThreadAssertions.h>
#include <wtf/WeakRef.h>

namespace WebCore {
class ImageTransferSessionVT;
class LocalSampleBufferDisplayLayer;
enum class VideoFrameRotation : uint16_t;
};

namespace WebKit {
class GPUConnectionToWebProcess;
struct SharedPreferencesForWebProcess;

class RemoteSampleBufferDisplayLayer : public RefCounted<RemoteSampleBufferDisplayLayer>, public WebCore::SampleBufferDisplayLayerClient, public IPC::MessageReceiver, private IPC::MessageSender {
    WTF_MAKE_TZONE_ALLOCATED(RemoteSampleBufferDisplayLayer);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static RefPtr<RemoteSampleBufferDisplayLayer> create(GPUConnectionToWebProcess&, SampleBufferDisplayLayerIdentifier, Ref<IPC::Connection>&&, RemoteSampleBufferDisplayLayerManager&);
    ~RemoteSampleBufferDisplayLayer();

    USING_CAN_MAKE_WEAKPTR(WebCore::SampleBufferDisplayLayerClient);

    using LayerInitializationCallback = CompletionHandler<void(std::optional<LayerHostingContextID>)>;
    void initialize(bool hideRootLayer, WebCore::IntSize, bool shouldMaintainAspectRatio, bool canShowWhileLocked, LayerInitializationCallback&&);

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

    CGRect bounds() const;
    void updateBoundsAndPosition(CGRect, std::optional<WTF::MachSendRight>&&);

    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const;

private:
    RemoteSampleBufferDisplayLayer(GPUConnectionToWebProcess&, SampleBufferDisplayLayerIdentifier, Ref<IPC::Connection>&&, RemoteSampleBufferDisplayLayerManager&);

    RefPtr<WebCore::LocalSampleBufferDisplayLayer> protectedSampleBufferDisplayLayer() const;

#if !RELEASE_LOG_DISABLED
    void setLogIdentifier(String&&);
#endif
    void updateDisplayMode(bool hideDisplayLayer, bool hideRootLayer);
    void flush();
    void flushAndRemoveImage();
    void play();
    void pause();
    void enqueueVideoFrame(SharedVideoFrame&&);
    void clearVideoFrames();
    void setSharedVideoFrameSemaphore(IPC::Semaphore&&);
    void setSharedVideoFrameMemory(WebCore::SharedMemory::Handle&&);
    void setShouldMaintainAspectRatio(bool shouldMaintainAspectRatio);

    // IPC::MessageSender
    IPC::Connection* messageSenderConnection() const final;
    uint64_t messageSenderDestinationID() const final { return m_identifier.toUInt64(); }

    // WebCore::SampleBufferDisplayLayerClient
    void sampleBufferDisplayLayerStatusDidFail() final;
#if PLATFORM(IOS_FAMILY)
    bool canShowWhileLocked() const final { return false; }
#endif

    ThreadSafeWeakPtr<GPUConnectionToWebProcess> m_gpuConnection WTF_GUARDED_BY_CAPABILITY(m_consumeThread);
    SampleBufferDisplayLayerIdentifier m_identifier;
    Ref<IPC::Connection> m_connection;
    RefPtr<WebCore::LocalSampleBufferDisplayLayer> m_sampleBufferDisplayLayer;
    std::unique_ptr<LayerHostingContext> m_layerHostingContext;
    SharedVideoFrameReader m_sharedVideoFrameReader;
    ThreadLikeAssertion m_consumeThread NO_UNIQUE_ADDRESS;
    ThreadSafeWeakPtr<RemoteSampleBufferDisplayLayerManager> m_remoteSampleBufferDisplayLayerManager;
};

}

#endif // PLATFORM(COCOA) && ENABLE(GPU_PROCESS) && ENABLE(MEDIA_STREAM)
