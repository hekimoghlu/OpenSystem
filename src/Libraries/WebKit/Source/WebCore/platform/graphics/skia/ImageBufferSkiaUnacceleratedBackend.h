/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 30, 2023.
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

#if USE(SKIA)

#include "ImageBuffer.h"
#include "ImageBufferSkiaSurfaceBackend.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class ImageBufferSkiaUnacceleratedBackend final : public ImageBufferSkiaSurfaceBackend {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(ImageBufferSkiaUnacceleratedBackend);
    WTF_MAKE_NONCOPYABLE(ImageBufferSkiaUnacceleratedBackend);
public:
    static std::unique_ptr<ImageBufferSkiaUnacceleratedBackend> create(const Parameters&, const ImageBufferCreationContext&);
    ~ImageBufferSkiaUnacceleratedBackend();

private:
    ImageBufferSkiaUnacceleratedBackend(const Parameters&, sk_sp<SkSurface>&&);

    RefPtr<NativeImage> copyNativeImage() final;
    RefPtr<NativeImage> createNativeImageReference() final;

    void getPixelBuffer(const IntRect&, PixelBuffer&) final;
    void putPixelBuffer(const PixelBuffer&, const IntRect& srcRect, const IntPoint& destPoint, AlphaPremultiplication destFormat) final;
};

} // namespace WebCore

#endif // USE(SKIA)
