/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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
#include "ImageBufferSet.h"
#include "PrepareBackingStoreBuffersData.h"
#include "RemoteImageBufferSetIdentifier.h"
#include "RenderingUpdateID.h"
#include "StreamConnectionWorkQueue.h"
#include "StreamMessageReceiver.h"
#include <WebCore/ImageBuffer.h>

#if ENABLE(RE_DYNAMIC_CONTENT_SCALING)
#include <WebCore/DynamicContentScalingDisplayList.h>
#endif

namespace WebKit {

class RemoteRenderingBackend;

class RemoteImageBufferSet : public IPC::StreamMessageReceiver, public ImageBufferSet {
public:
    static Ref<RemoteImageBufferSet> create(RemoteImageBufferSetIdentifier, WebCore::RenderingResourceIdentifier displayListIdentifier, RemoteRenderingBackend&);
    ~RemoteImageBufferSet();
    void stopListeningForIPC();

    // Ensures frontBuffer is valid, either by swapping an existing back
    // buffer, or allocating a new one.
    void ensureBufferForDisplay(ImageBufferSetPrepareBufferForDisplayInputData&, SwapBuffersDisplayRequirement&, bool isSync);

    // Initializes the contents of the new front buffer using the previous
    // frames (if applicable), clips to the dirty region, and clears the pixels
    // to be drawn (unless drawing will be opaque).
    void prepareBufferForDisplay(const WebCore::Region& dirtyRegion, bool requiresClearedPixels);

    bool makeBuffersVolatile(OptionSet<BufferInSetType> requestedBuffers, OptionSet<BufferInSetType>& volatileBuffers, bool forcePurge);

private:
    RemoteImageBufferSet(RemoteImageBufferSetIdentifier, WebCore::RenderingResourceIdentifier, RemoteRenderingBackend&);
    void startListeningForIPC();
    IPC::StreamConnectionWorkQueue& workQueue() const;

    // IPC::StreamMessageReceiver
    void didReceiveStreamMessage(IPC::StreamServerConnection&, IPC::Decoder&) final;

    // Messages
    void updateConfiguration(const WebCore::FloatSize&, WebCore::RenderingMode, WebCore::RenderingPurpose, float resolutionScale, const WebCore::DestinationColorSpace&, WebCore::ImageBufferPixelFormat);
    void endPrepareForDisplay(RenderingUpdateID);

#if ENABLE(RE_DYNAMIC_CONTENT_SCALING)
    void dynamicContentScalingDisplayList(CompletionHandler<void(std::optional<WebCore::DynamicContentScalingDisplayList>&&)>&&);
    WebCore::DynamicContentScalingResourceCache ensureDynamicContentScalingResourceCache();
#endif

    bool isOpaque() const
    {
#if HAVE(IOSURFACE_RGB10)
        if (m_pixelFormat == WebCore::ImageBufferPixelFormat::RGB10)
            return true;
#endif
        return m_pixelFormat == WebCore::ImageBufferPixelFormat::BGRX8;
    }

    const RemoteImageBufferSetIdentifier m_identifier;
    const WebCore::RenderingResourceIdentifier m_displayListIdentifier;
    RefPtr<RemoteRenderingBackend> m_backend;

    WebCore::FloatSize m_logicalSize;
    WebCore::RenderingMode m_renderingMode;
    WebCore::RenderingPurpose m_renderingPurpose;
    float m_resolutionScale { 1.0f };
    WebCore::DestinationColorSpace m_colorSpace { WebCore::DestinationColorSpace::SRGB() };
    WebCore::ImageBufferPixelFormat m_pixelFormat;
    bool m_displayListCreated { false };

    std::optional<WebCore::IntRect> m_previouslyPaintedRect;

#if ENABLE(RE_DYNAMIC_CONTENT_SCALING)
    WebCore::DynamicContentScalingResourceCache m_dynamicContentScalingResourceCache;
#endif
};


} // namespace WebKit

#endif
