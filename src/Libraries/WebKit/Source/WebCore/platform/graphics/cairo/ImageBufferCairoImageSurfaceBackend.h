/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 12, 2025.
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

#include "ImageBufferCairoSurfaceBackend.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class ImageBufferCairoImageSurfaceBackend : public ImageBufferCairoSurfaceBackend {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ImageBufferCairoImageSurfaceBackend);
    WTF_MAKE_NONCOPYABLE(ImageBufferCairoImageSurfaceBackend);
public:
    static IntSize calculateSafeBackendSize(const Parameters&);
    static unsigned calculateBytesPerRow(const IntSize& backendSize);
    static size_t calculateMemoryCost(const Parameters&);

    static std::unique_ptr<ImageBufferCairoImageSurfaceBackend> create(const Parameters&, const ImageBufferCreationContext&);
    static std::unique_ptr<ImageBufferCairoImageSurfaceBackend> create(const Parameters&, const GraphicsContext&);

private:
    ImageBufferCairoImageSurfaceBackend(const Parameters&, RefPtr<cairo_surface_t>&&);

    unsigned bytesPerRow() const override;

    void platformTransformColorSpace(const std::array<uint8_t, 256>& lookUpTable) override;
};

} // namespace WebCore

#endif // USE(CAIRO)
