/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 24, 2025.
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

#if ENABLE(RE_DYNAMIC_CONTENT_SCALING)

#include "ImageBufferBackendHandleSharing.h"
#include <WebCore/ImageBuffer.h>
#include <WebCore/ImageBufferCGBackend.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {

class DynamicContentScalingImageBufferBackend : public WebCore::ImageBufferCGBackend, public ImageBufferBackendHandleSharing {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DynamicContentScalingImageBufferBackend);
    WTF_MAKE_NONCOPYABLE(DynamicContentScalingImageBufferBackend);
public:
    static size_t calculateMemoryCost(const Parameters&);

    static std::unique_ptr<DynamicContentScalingImageBufferBackend> create(const Parameters&, const WebCore::ImageBufferCreationContext&);

    DynamicContentScalingImageBufferBackend(const Parameters&, const WebCore::ImageBufferCreationContext&, WebCore::RenderingMode);
    ~DynamicContentScalingImageBufferBackend();

    WebCore::GraphicsContext& context() final;
    std::optional<ImageBufferBackendHandle> createBackendHandle(WebCore::SharedMemory::Protection = WebCore::SharedMemory::Protection::ReadWrite) const final;

    void releaseGraphicsContext() final;

    bool canMapBackingStore() const final;

    // NOTE: These all ASSERT_NOT_REACHED().
    RefPtr<WebCore::NativeImage> copyNativeImage() final;
    RefPtr<WebCore::NativeImage> createNativeImageReference() final;
    void getPixelBuffer(const WebCore::IntRect&, WebCore::PixelBuffer&) final;
    void putPixelBuffer(const WebCore::PixelBuffer&, const WebCore::IntRect& srcRect, const WebCore::IntPoint& destPoint, WebCore::AlphaPremultiplication destFormat) final;


protected:
    unsigned bytesPerRow() const final;
    String debugDescription() const final;

    // ImageBufferBackendSharing
    ImageBufferBackendSharing* toBackendSharing() final { return this; }

    mutable std::unique_ptr<WebCore::GraphicsContextCG> m_context;
    RetainPtr<id> m_resourceCache;
    WebCore::RenderingMode m_renderingMode;
};

}

#endif // ENABLE(RE_DYNAMIC_CONTENT_SCALING)
