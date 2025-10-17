/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 25, 2023.
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

#include "IPCEvent.h"
#include "ScopedRenderingResourcesRequest.h"
#include "StreamMessageReceiver.h"
#include <WebCore/ImageBuffer.h>
#include <WebCore/ShareableBitmap.h>

#if ENABLE(RE_DYNAMIC_CONTENT_SCALING)
#include <WebCore/DynamicContentScalingDisplayList.h>
#endif

namespace IPC {
class Semaphore;
class StreamConnectionWorkQueue;
}

namespace WebKit {

class RemoteRenderingBackend;

class RemoteImageBuffer : public IPC::StreamMessageReceiver {
public:
    static Ref<RemoteImageBuffer> create(Ref<WebCore::ImageBuffer>, RemoteRenderingBackend&);
    ~RemoteImageBuffer();
    void stopListeningForIPC();
    WebCore::RenderingResourceIdentifier identifier() const { return m_imageBuffer->renderingResourceIdentifier(); }
    Ref<WebCore::ImageBuffer> imageBuffer() const { return m_imageBuffer; }
private:
    RemoteImageBuffer(Ref<WebCore::ImageBuffer>, RemoteRenderingBackend&);
    void startListeningForIPC();
    IPC::StreamConnectionWorkQueue& workQueue() const;

    // IPC::StreamMessageReceiver
    void didReceiveStreamMessage(IPC::StreamServerConnection&, IPC::Decoder&) final;

    // Messages
    // This is using location and size as opposed to rect because invalid rect object gets restricted by IPC rect object decoder and triggers timeout in WebContent process.
    void getPixelBuffer(WebCore::PixelBufferFormat, WebCore::IntPoint srcPoint, WebCore::IntSize srcSize, CompletionHandler<void()>&&);
    void getPixelBufferWithNewMemory(WebCore::SharedMemory::Handle&&, WebCore::PixelBufferFormat, WebCore::IntPoint srcPoint, WebCore::IntSize srcSize, CompletionHandler<void()>&&);
    void putPixelBuffer(Ref<WebCore::PixelBuffer>, WebCore::IntPoint srcPoint, WebCore::IntSize srcSize, WebCore::IntPoint destPoint, WebCore::AlphaPremultiplication destFormat);
    void getShareableBitmap(WebCore::PreserveResolution, CompletionHandler<void(std::optional<WebCore::ShareableBitmap::Handle>&&)>&&);
    void filteredNativeImage(Ref<WebCore::Filter>, CompletionHandler<void(std::optional<WebCore::ShareableBitmap::Handle>&&)>&&);
    void convertToLuminanceMask();
    void transformToColorSpace(const WebCore::DestinationColorSpace&);
    void flushContext();
    void flushContextSync(CompletionHandler<void()>&&);

#if ENABLE(RE_DYNAMIC_CONTENT_SCALING)
    void dynamicContentScalingDisplayList(CompletionHandler<void(std::optional<WebCore::DynamicContentScalingDisplayList>&&)>&&);
#endif

    RefPtr<RemoteRenderingBackend> m_backend;
    Ref<WebCore::ImageBuffer> m_imageBuffer;
    ScopedRenderingResourcesRequest m_renderingResourcesRequest { ScopedRenderingResourcesRequest::acquire() };
};

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
