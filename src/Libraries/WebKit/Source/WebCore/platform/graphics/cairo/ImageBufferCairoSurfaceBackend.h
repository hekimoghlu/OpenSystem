/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 15, 2022.
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

#if USE(CAIRO)

#include "GraphicsContextCairo.h"
#include "ImageBufferCairoBackend.h"

namespace WebCore {

class ImageBufferCairoSurfaceBackend : public ImageBufferCairoBackend {
public:
    GraphicsContext& context() override;

    RefPtr<NativeImage> copyNativeImage() override;
    RefPtr<NativeImage> createNativeImageReference() override;

    bool canMapBackingStore() const final;
    RefPtr<cairo_surface_t> createCairoSurface() override;
    void getPixelBuffer(const IntRect&, PixelBuffer&) override;
    void putPixelBuffer(const PixelBuffer&, const IntRect& srcRect, const IntPoint& destPoint, AlphaPremultiplication destFormat) override;

protected:
    ImageBufferCairoSurfaceBackend(const Parameters&, RefPtr<cairo_surface_t>&&);

    RefPtr<NativeImage> cairoSurfaceCoerceToImage();
    unsigned bytesPerRow() const override;
    String debugDescription() const override;

    RefPtr<cairo_surface_t> m_surface;
    mutable GraphicsContextCairo m_context;
};

} // namespace WebCore

#endif // USE(CAIRO)
