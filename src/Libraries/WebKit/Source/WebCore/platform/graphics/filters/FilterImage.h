/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 5, 2023.
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

#include "FloatRect.h"
#include "ImageBuffer.h"
#include "IntRect.h"
#include "PixelBuffer.h"
#include "RenderingMode.h"
#include <JavaScriptCore/Forward.h>
#include <wtf/RefCounted.h>
#include <wtf/Vector.h>

#if USE(CORE_IMAGE)
OBJC_CLASS CIImage;
#endif

#if USE(SKIA)
#include <skia/core/SkPicture.h>
#include <skia/core/SkPictureRecorder.h>
#endif

namespace WebCore {

class Filter;
class FloatRect;

class FilterImage : public RefCounted<FilterImage> {
public:
    static RefPtr<FilterImage> create(const FloatRect& primitiveSubregion, const FloatRect& imageRect, const IntRect& absoluteImageRect, bool isAlphaImage, bool isValidPremultiplied, RenderingMode, const DestinationColorSpace&, ImageBufferAllocator&);
    static RefPtr<FilterImage> create(const FloatRect& primitiveSubregion, const FloatRect& imageRect, const IntRect& absoluteImageRect, Ref<ImageBuffer>&&, ImageBufferAllocator&);

    // The return values are in filter coordinates.
    FloatRect primitiveSubregion() const { return m_primitiveSubregion; }
    FloatRect maxEffectRect(const Filter&) const;
    FloatRect imageRect() const { return m_imageRect; }

    // The return values are in user-space coordinates.
    IntRect absoluteImageRect() const { return m_absoluteImageRect; }
    IntRect absoluteImageRectRelativeTo(const FilterImage& origin) const;
    FloatPoint mappedAbsolutePoint(const FloatPoint&) const;

    bool isAlphaImage() const { return m_isAlphaImage; }
    RenderingMode renderingMode() const { return m_renderingMode; }
    const DestinationColorSpace& colorSpace() const { return m_colorSpace; }

    size_t memoryCost() const;

    WEBCORE_EXPORT ImageBuffer* imageBuffer();
    PixelBuffer* pixelBuffer(AlphaPremultiplication);

    RefPtr<PixelBuffer> getPixelBuffer(AlphaPremultiplication, const IntRect& sourceRect, std::optional<DestinationColorSpace> = std::nullopt);
    void copyPixelBuffer(PixelBuffer& destinationPixelBuffer, const IntRect& sourceRect);

    void correctPremultipliedPixelBuffer();
    void transformToColorSpace(const DestinationColorSpace&);

#if USE(CORE_IMAGE)
    RetainPtr<CIImage> ciImage() const { return m_ciImage; }
    void setCIImage(RetainPtr<CIImage>&&);
    size_t memoryCostOfCIImage() const;
#endif

private:
    FilterImage(const FloatRect& primitiveSubregion, const FloatRect& imageRect, const IntRect& absoluteImageRect, bool isAlphaImage, bool isValidPremultiplied, RenderingMode, const DestinationColorSpace&, ImageBufferAllocator&);
    FilterImage(const FloatRect& primitiveSubregion, const FloatRect& imageRect, const IntRect& absoluteImageRect, Ref<ImageBuffer>&&, ImageBufferAllocator&);

    RefPtr<PixelBuffer>& pixelBufferSlot(AlphaPremultiplication);

    ImageBuffer* imageBufferFromPixelBuffer();

#if USE(CORE_IMAGE)
    ImageBuffer* imageBufferFromCIImage();
#endif

    bool requiresPixelBufferColorSpaceConversion(std::optional<DestinationColorSpace>) const;

    FloatRect m_primitiveSubregion;
    FloatRect m_imageRect;
    IntRect m_absoluteImageRect;

    bool m_isAlphaImage { false };
    bool m_isValidPremultiplied { true };
    RenderingMode m_renderingMode;
    DestinationColorSpace m_colorSpace;

    RefPtr<ImageBuffer> m_imageBuffer;
    RefPtr<PixelBuffer> m_unpremultipliedPixelBuffer;
    RefPtr<PixelBuffer> m_premultipliedPixelBuffer;

#if USE(CORE_IMAGE)
    RetainPtr<CIImage> m_ciImage;
#endif

    ImageBufferAllocator& m_allocator;
};

} // namespace WebCore
