/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 23, 2023.
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

#include "Color.h"
#include "IntRect.h"
#include "IntSize.h"
#include "NativeImage.h"
#include "SharedBuffer.h"
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/ParsingUtilities.h>

namespace WebCore {

#if USE(CAIRO)
// Due to the pixman 16.16 floating point representation, cairo is not able to handle
// images whose size is bigger than 32768.
static const int cairoMaxImageSize = 32768;
#endif

class ImageBackingStore {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(ImageBackingStore);
public:
    static std::unique_ptr<ImageBackingStore> create(const IntSize& size, bool premultiplyAlpha = true)
    {
        return std::unique_ptr<ImageBackingStore>(new ImageBackingStore(size, premultiplyAlpha));
    }

    static std::unique_ptr<ImageBackingStore> create(const ImageBackingStore& other)
    {
        return std::unique_ptr<ImageBackingStore>(new ImageBackingStore(other));
    }

    PlatformImagePtr image() const;

    bool setSize(const IntSize& size)
    {
        if (size.isEmpty())
            return false;

        Vector<uint8_t> buffer;
        size_t bufferSize = size.area() * sizeof(uint32_t);

        if (!buffer.tryReserveCapacity(bufferSize))
            return false;

        buffer.grow(bufferSize);
        m_pixels = FragmentedSharedBuffer::DataSegment::create(WTFMove(buffer));
        m_pixelsSpan = spanReinterpretCast<uint32_t>(spanConstCast<uint8_t>(m_pixels->span()));
        m_size = size;
        m_frameRect = IntRect(IntPoint(), m_size);
        clear();
        return true;
    }

    void setFrameRect(const IntRect& frameRect)
    {
        ASSERT(!m_size.isEmpty());
        ASSERT(inBounds(frameRect));
        m_frameRect = frameRect;
    }

    const IntSize& size() const { return m_size; }
    const IntRect& frameRect() const { return m_frameRect; }

    void clear()
    {
        zeroSpan(m_pixelsSpan);
    }

    void clearRect(const IntRect& rect)
    {
        if (rect.isEmpty() || !inBounds(rect))
            return;

        auto pixels = pixelsStartingAt(rect.x(), rect.y());
        for (int i = 0; i < rect.height(); ++i) {
            zeroSpan(pixels.first(rect.width()));
            skip(pixels, m_size.width());
        }
    }

    void fillRect(const IntRect& rect, uint8_t r, uint8_t g, uint8_t b, uint8_t a)
    {
        if (rect.isEmpty() || !inBounds(rect))
            return;

        auto pixels = pixelsStartingAt(rect.x(), rect.y());
        uint32_t pixelValue = this->pixelValue(r, g, b, a);
        for (int i = 0; i < rect.height(); ++i) {
            for (int j = 0; j < rect.width(); ++j)
                pixels[j] = pixelValue;
            skip(pixels, m_size.width());
        }
    }

    void repeatFirstRow(const IntRect& rect)
    {
        if (rect.isEmpty() || !inBounds(rect))
            return;

        auto sourcePixels = pixelsStartingAt(rect.x(), rect.y());
        auto destinationPixels = sourcePixels.subspan(m_size.width());
        auto sourceRow = sourcePixels.first(rect.width());
        for (int i = 1; i < rect.height(); ++i) {
            memcpySpan(destinationPixels, sourceRow);
            skip(destinationPixels, m_size.width());
        }
    }

    std::span<uint32_t> pixelsStartingAt(int x, int y)
    {
        ASSERT(inBounds(IntPoint(x, y)));
        return m_pixelsSpan.subspan(y * m_size.width() + x);
    }

    uint32_t& pixelAt(int x, int y)
    {
        ASSERT(inBounds(IntPoint(x, y)));
        return m_pixelsSpan[y * m_size.width() + x];
    }

    void setPixel(uint32_t& destination, uint8_t r, uint8_t g, uint8_t b, uint8_t a)
    {
        destination = pixelValue(r, g, b, a);
    }

    void setPixel(int x, int y, uint8_t r, uint8_t g, uint8_t b, uint8_t a)
    {
        setPixel(pixelAt(x, y), r, g, b, a);
    }

    void blendPixel(uint32_t& destination, uint8_t r, uint8_t g, uint8_t b, uint8_t a)
    {
        if (!a)
            return;

        auto pixel = asSRGBA(PackedColor::ARGB { destination }).resolved();

        if (a >= 255 || !pixel.alpha) {
            setPixel(destination, r, g, b, a);
            return;
        }

        if (!m_premultiplyAlpha)
            pixel = premultipliedFlooring(pixel).resolved();

        uint8_t d = 255 - a;

        r = fastDivideBy255(r * a + pixel.red * d);
        g = fastDivideBy255(g * a + pixel.green * d);
        b = fastDivideBy255(b * a + pixel.blue * d);
        a += fastDivideBy255(d * pixel.alpha);

        auto result = SRGBA<uint8_t> { r, g, b, a };

        if (!m_premultiplyAlpha)
            result = unpremultiplied(result);

        destination = PackedColor::ARGB { result }.value;
    }

    static bool isOverSize(const IntSize& size)
    {
#if USE(CAIRO)
        // FIXME: this is a workaround to avoid the cairo image size limit, but we should implement support for
        // bigger images. See https://bugs.webkit.org/show_bug.cgi?id=177227.
        //
        // If the image is bigger than the cairo limit it can't be displayed, so we don't even try to decode it.
        if (size.width() > cairoMaxImageSize || size.height() > cairoMaxImageSize)
            return true;
#endif
        static unsigned long long MaxPixels = ((1 << 29) - 1);
        unsigned long long pixels = static_cast<unsigned long long>(size.width()) * static_cast<unsigned long long>(size.height());
        return pixels > MaxPixels;
    }

private:
    ImageBackingStore(const IntSize& size, bool premultiplyAlpha = true)
        : m_premultiplyAlpha(premultiplyAlpha)
    {
        ASSERT(!size.isEmpty() && !isOverSize(size));
        setSize(size);
    }

    ImageBackingStore(const ImageBackingStore& other)
        : m_size(other.m_size)
        , m_premultiplyAlpha(other.m_premultiplyAlpha)
    {
        ASSERT(!m_size.isEmpty() && !isOverSize(m_size));
        Vector<uint8_t> buffer(other.m_pixels->span());
        m_pixels = FragmentedSharedBuffer::DataSegment::create(WTFMove(buffer));
        m_pixelsSpan = spanReinterpretCast<uint32_t>(spanConstCast<uint8_t>(m_pixels->span()));
    }

    bool inBounds(const IntPoint& point) const
    {
        return IntRect(IntPoint(), m_size).contains(point);
    }

    bool inBounds(const IntRect& rect) const
    {
        return IntRect(IntPoint(), m_size).contains(rect);
    }

    uint32_t pixelValue(uint8_t r, uint8_t g, uint8_t b, uint8_t a) const
    {
        if (m_premultiplyAlpha && !a)
            return 0;

        auto result = SRGBA<uint8_t> { r, g, b, a };

        if (m_premultiplyAlpha && a < 255)
            result = premultipliedFlooring(result);

        return PackedColor::ARGB { result }.value;
    }

    // m_pixels type should be identical to the one set in ImageBackingStoreCairo.cpp
    RefPtr<FragmentedSharedBuffer::DataSegment> m_pixels;
    std::span<uint32_t> m_pixelsSpan;
    IntSize m_size;
    IntRect m_frameRect; // This will always just be the entire buffer except for GIF and PNG frames
    bool m_premultiplyAlpha { true };
};

}
