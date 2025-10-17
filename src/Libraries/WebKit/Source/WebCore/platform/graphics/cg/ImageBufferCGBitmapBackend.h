/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 16, 2022.
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

class ImageBufferCGBitmapBackend final : public ImageBufferCGBackend {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ImageBufferCGBitmapBackend);
    WTF_MAKE_NONCOPYABLE(ImageBufferCGBitmapBackend);
public:
    ~ImageBufferCGBitmapBackend();

    static size_t calculateMemoryCost(const Parameters&);

    static std::unique_ptr<ImageBufferCGBitmapBackend> create(const Parameters&, const ImageBufferCreationContext&);
    bool canMapBackingStore() const final;
    GraphicsContext& context() final;

private:
    ImageBufferCGBitmapBackend(const Parameters&, std::span<uint8_t> data, RetainPtr<CGDataProviderRef>&&, std::unique_ptr<GraphicsContextCG>&&);

    unsigned bytesPerRow() const final;

    RefPtr<NativeImage> copyNativeImage() final;
    RefPtr<NativeImage> createNativeImageReference() final;

    void getPixelBuffer(const IntRect&, PixelBuffer&) final;
    void putPixelBuffer(const PixelBuffer&, const IntRect& srcRect, const IntPoint& destPoint, AlphaPremultiplication destFormat) final;

    std::span<uint8_t> m_data;
    RetainPtr<CGDataProviderRef> m_dataProvider;
};

} // namespace WebCore

#endif // USE(CG)
