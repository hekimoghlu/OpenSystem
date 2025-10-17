/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 9, 2024.
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

#if USE(CG)

#include "ImageBuffer.h"
#include "ImageBufferCGBackend.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class ImageBufferCGPDFDocumentBackend : public ImageBufferCGBackend {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ImageBufferCGPDFDocumentBackend);
    WTF_MAKE_NONCOPYABLE(ImageBufferCGPDFDocumentBackend);
public:
    WEBCORE_EXPORT static size_t calculateMemoryCost(const Parameters&);
    WEBCORE_EXPORT static std::unique_ptr<ImageBufferCGPDFDocumentBackend> create(const Parameters&, const ImageBufferCreationContext&);

    ~ImageBufferCGPDFDocumentBackend();

    static constexpr RenderingMode renderingMode = RenderingMode::PDFDocument;

private:
    ImageBufferCGPDFDocumentBackend(const Parameters&, RetainPtr<CFDataRef>&&, std::unique_ptr<GraphicsContextCG>&&);

    bool canMapBackingStore() const { return false; }
    unsigned bytesPerRow() const final { return 0; }
    GraphicsContext& context() final;

    RefPtr<NativeImage> copyNativeImage() final { return createNativeImageReference(); }
    RefPtr<NativeImage> createNativeImageReference() final { return nullptr; }

    void getPixelBuffer(const IntRect&, PixelBuffer&) final { ASSERT_NOT_REACHED(); }
    void putPixelBuffer(const PixelBuffer&, const IntRect&, const IntPoint&, AlphaPremultiplication) final { ASSERT_NOT_REACHED(); }

    RefPtr<SharedBuffer> sinkIntoPDFDocument() final;

    String debugDescription() const final;

    RetainPtr<CFDataRef> m_data;
};

} // namespace WebCore

#endif // USE(CG)
