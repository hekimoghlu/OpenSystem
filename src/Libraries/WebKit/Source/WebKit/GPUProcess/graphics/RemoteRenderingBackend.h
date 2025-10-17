/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 17, 2022.
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

#if ENABLE(GPU_PROCESS)

#include "BufferIdentifierSet.h"
#include "Connection.h"
#include "IPCEvent.h"
#include "ImageBufferBackendHandle.h"
#include "MarkSurfacesAsVolatileRequestIdentifier.h"
#include "MessageReceiver.h"
#include "MessageSender.h"
#include "RemoteImageBufferSetIdentifier.h"
#include "RemoteResourceCache.h"
#include "RemoteSerializedImageBufferIdentifier.h"
#include "RemoteSharedResourceCache.h"
#include "RenderingBackendIdentifier.h"
#include "RenderingUpdateID.h"
#include "ScopedActiveMessageReceiveQueue.h"
#include "ScopedRenderingResourcesRequest.h"
#include "ShapeDetectionIdentifier.h"
#include "StreamConnectionWorkQueue.h"
#include "StreamMessageReceiver.h"
#include "StreamServerConnection.h"
#include <WebCore/ImageBufferPixelFormat.h>
#include <WebCore/ProcessIdentity.h>
#include <WebCore/RenderingResourceIdentifier.h>
#include <wtf/HashMap.h>
#include <wtf/WeakPtr.h>

namespace WTF {
enum class Critical : bool;
enum class Synchronous : bool;
}

namespace WebCore {
class DestinationColorSpace;
class FloatSize;
class MediaPlayer;
class NativeImage;
enum class RenderingMode : uint8_t;
struct FontCustomPlatformSerializedData;

namespace ShapeDetection {
struct BarcodeDetectorOptions;
enum class BarcodeFormat : uint8_t;
struct FaceDetectorOptions;
}

}

namespace WebKit {

class GPUConnectionToWebProcess;
class RemoteDisplayListRecorder;
class RemoteImageBuffer;
class RemoteImageBufferSet;
struct BufferIdentifierSet;
struct ImageBufferSetPrepareBufferForDisplayInputData;
struct ImageBufferSetPrepareBufferForDisplayOutputData;
struct SharedPreferencesForWebProcess;
enum class SwapBuffersDisplayRequirement : uint8_t;

namespace ShapeDetection {
class ObjectHeap;
}

class RemoteRenderingBackend : private IPC::MessageSender, public IPC::StreamMessageReceiver, public CanMakeWeakPtr<RemoteRenderingBackend> {
public:
    static Ref<RemoteRenderingBackend> create(GPUConnectionToWebProcess&, RenderingBackendIdentifier, Ref<IPC::StreamServerConnection>&&);
    virtual ~RemoteRenderingBackend();
    void stopListeningForIPC();

    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() const;

    RemoteResourceCache& remoteResourceCache() { return m_remoteResourceCache; }
    RemoteSharedResourceCache& sharedResourceCache() { return m_sharedResourceCache; }
    Ref<RemoteSharedResourceCache> protectedSharedResourceCache() { return m_sharedResourceCache; }

    void didReceiveStreamMessage(IPC::StreamServerConnection&, IPC::Decoder&) final;

    // Runs Function in RemoteRenderingBackend task queue.
    void dispatch(Function<void()>&&);

    IPC::StreamServerConnection& streamConnection() const { return m_streamConnection.get(); }
    Ref<IPC::StreamServerConnection> protectedStreamConnection() const { return m_streamConnection; }

    GPUConnectionToWebProcess& gpuConnectionToWebProcess() { return m_gpuConnectionToWebProcess.get(); }
    Ref<GPUConnectionToWebProcess> protectedGPUConnectionToWebProcess();

    void setSharedMemoryForGetPixelBuffer(RefPtr<WebCore::SharedMemory> memory) { m_getPixelBufferSharedMemory = WTFMove(memory); }
    RefPtr<WebCore::SharedMemory> sharedMemoryForGetPixelBuffer() const { return m_getPixelBufferSharedMemory; }

    IPC::StreamConnectionWorkQueue& workQueue() const { return m_workQueue; }
    Ref<IPC::StreamConnectionWorkQueue> protectedWorkQueue() const { return m_workQueue; }

    RefPtr<WebCore::ImageBuffer> imageBuffer(WebCore::RenderingResourceIdentifier);
    RefPtr<WebCore::ImageBuffer> takeImageBuffer(WebCore::RenderingResourceIdentifier);

    RefPtr<WebCore::ImageBuffer> allocateImageBuffer(const WebCore::FloatSize& logicalSize, WebCore::RenderingMode, WebCore::RenderingPurpose, float resolutionScale, const WebCore::DestinationColorSpace&, WebCore::ImageBufferPixelFormat, WebCore::ImageBufferCreationContext, WebCore::RenderingResourceIdentifier);

    void terminateWebProcess(ASCIILiteral message);

    RenderingBackendIdentifier identifier() { return m_renderingBackendIdentifier; }
private:
    friend class RemoteImageBufferSet;
    RemoteRenderingBackend(GPUConnectionToWebProcess&, RenderingBackendIdentifier, Ref<IPC::StreamServerConnection>&&);
    void startListeningForIPC();
    void workQueueInitialize();
    void workQueueUninitialize();

    // IPC::MessageSender.
    IPC::Connection* messageSenderConnection() const override;
    uint64_t messageSenderDestinationID() const override;

    // Messages to be received.
    void createImageBuffer(const WebCore::FloatSize& logicalSize, WebCore::RenderingMode, WebCore::RenderingPurpose, float resolutionScale, const WebCore::DestinationColorSpace&, WebCore::ImageBufferPixelFormat, WebCore::RenderingResourceIdentifier);
    void releaseImageBuffer(WebCore::RenderingResourceIdentifier);
    void moveToSerializedBuffer(WebCore::RenderingResourceIdentifier);
    void moveToImageBuffer(WebCore::RenderingResourceIdentifier);
#if PLATFORM(COCOA)
    void didDrawRemoteToPDF(WebCore::PageIdentifier, WebCore::RenderingResourceIdentifier, WebCore::SnapshotIdentifier);
#endif
    void destroyGetPixelBufferSharedMemory();
    void cacheNativeImage(WebCore::ShareableBitmap::Handle&&, WebCore::RenderingResourceIdentifier);
    void cacheDecomposedGlyphs(Ref<WebCore::DecomposedGlyphs>&&);
    void cacheGradient(Ref<WebCore::Gradient>&&);
    void cacheFilter(Ref<WebCore::Filter>&&);
    void cacheFont(const WebCore::Font::Attributes&, WebCore::FontPlatformDataAttributes, std::optional<WebCore::RenderingResourceIdentifier>);
    void cacheFontCustomPlatformData(WebCore::FontCustomPlatformSerializedData&&);
    void releaseAllDrawingResources();
    void releaseAllImageResources();
    void releaseRenderingResource(WebCore::RenderingResourceIdentifier);
    void finalizeRenderingUpdate(RenderingUpdateID);
    void markSurfacesVolatile(MarkSurfacesAsVolatileRequestIdentifier, const Vector<std::pair<RemoteImageBufferSetIdentifier, OptionSet<BufferInSetType>>>&, bool forcePurge);
    void createRemoteImageBufferSet(WebKit::RemoteImageBufferSetIdentifier, WebCore::RenderingResourceIdentifier displayListIdentifier);
    void releaseRemoteImageBufferSet(WebKit::RemoteImageBufferSetIdentifier);

#if USE(GRAPHICS_LAYER_WC)
    void flush(IPC::Semaphore&&);
#endif

#if PLATFORM(COCOA)
    void prepareImageBufferSetsForDisplay(Vector<ImageBufferSetPrepareBufferForDisplayInputData> swapBuffersInput);
    void prepareImageBufferSetsForDisplaySync(Vector<ImageBufferSetPrepareBufferForDisplayInputData> swapBuffersInput, CompletionHandler<void(Vector<SwapBuffersDisplayRequirement>&&)>&&);

#endif

    void createRemoteBarcodeDetector(ShapeDetectionIdentifier, const WebCore::ShapeDetection::BarcodeDetectorOptions&);
    void releaseRemoteBarcodeDetector(ShapeDetectionIdentifier);
    void getRemoteBarcodeDetectorSupportedFormats(CompletionHandler<void(Vector<WebCore::ShapeDetection::BarcodeFormat>&&)>&&);
    void createRemoteFaceDetector(ShapeDetectionIdentifier, const WebCore::ShapeDetection::FaceDetectorOptions&);
    void releaseRemoteFaceDetector(ShapeDetectionIdentifier);
    void createRemoteTextDetector(ShapeDetectionIdentifier);
    void releaseRemoteTextDetector(ShapeDetectionIdentifier);

    void didFailCreateImageBuffer(WebCore::RenderingResourceIdentifier) WTF_REQUIRES_CAPABILITY(workQueue());
    void didCreateImageBuffer(Ref<WebCore::ImageBuffer>) WTF_REQUIRES_CAPABILITY(workQueue());

    void createDisplayListRecorder(RefPtr<WebCore::ImageBuffer>, WebCore::RenderingResourceIdentifier);
    void releaseDisplayListRecorder(WebCore::RenderingResourceIdentifier);

    Ref<ShapeDetection::ObjectHeap> protectedShapeDetectionObjectHeap() const;

    bool shouldUseLockdownFontParser() const;

    void getImageBufferResourceLimitsForTesting(CompletionHandler<void(WebCore::ImageBufferResourceLimits)>&&);

    const Ref<IPC::StreamConnectionWorkQueue> m_workQueue;
    const Ref<IPC::StreamServerConnection> m_streamConnection;
    const Ref<GPUConnectionToWebProcess> m_gpuConnectionToWebProcess;
    Ref<RemoteSharedResourceCache> m_sharedResourceCache;
    RemoteResourceCache m_remoteResourceCache;
    WebCore::ProcessIdentity m_resourceOwner;
    RenderingBackendIdentifier m_renderingBackendIdentifier;
    RefPtr<WebCore::SharedMemory> m_getPixelBufferSharedMemory;

    HashMap<WebCore::RenderingResourceIdentifier, IPC::ScopedActiveMessageReceiveQueue<RemoteDisplayListRecorder>> m_remoteDisplayLists WTF_GUARDED_BY_CAPABILITY(workQueue());
    HashMap<WebCore::RenderingResourceIdentifier, IPC::ScopedActiveMessageReceiveQueue<RemoteImageBuffer>> m_remoteImageBuffers WTF_GUARDED_BY_CAPABILITY(workQueue());
    HashMap<WebKit::RemoteImageBufferSetIdentifier, IPC::ScopedActiveMessageReceiveQueue<RemoteImageBufferSet>> m_remoteImageBufferSets WTF_GUARDED_BY_CAPABILITY(workQueue());
    Ref<ShapeDetection::ObjectHeap> m_shapeDetectionObjectHeap;
};

bool isSmallLayerBacking(const WebCore::ImageBufferParameters&);

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
