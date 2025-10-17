/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 22, 2023.
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
#include "config.h"
#include "PixelBuffer.h"

#include <JavaScriptCore/TypedArrayInlines.h>
#include <wtf/StdLibExtras.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

bool PixelBuffer::supportedPixelFormat(PixelFormat pixelFormat)
{
    switch (pixelFormat) {
    case PixelFormat::RGBA8:
    case PixelFormat::BGRA8:
#if HAVE(HDR_SUPPORT)
    case PixelFormat::RGBA16F:
#endif
        return true;

    case PixelFormat::BGRX8:
#if HAVE(IOSURFACE_RGB10)
    case PixelFormat::RGB10:
    case PixelFormat::RGB10A8:
#endif
        return false;
    }

    ASSERT_NOT_REACHED();
    return false;
}

static CheckedUint32 mustFitInInt32(CheckedUint32 uint32)
{
    if (!uint32.hasOverflowed() && !isInBounds<int32_t>(uint32.value()))
        uint32.overflowed();
    return uint32;
}

static CheckedUint32 computeRawPixelCount(const IntSize& size)
{
    return CheckedUint32 { size.width() } * size.height();
}

static CheckedUint32 computeRawPixelComponentCount(PixelFormat pixelFormat, const IntSize& size)
{
    ASSERT_UNUSED(pixelFormat, PixelBuffer::supportedPixelFormat(pixelFormat));
    constexpr unsigned componentsPerPixel = 4;
    return computeRawPixelCount(size) * componentsPerPixel;
}

CheckedUint32 PixelBuffer::computePixelCount(const IntSize& size)
{
    return mustFitInInt32(computeRawPixelCount(size));
}

CheckedUint32 PixelBuffer::computePixelComponentCount(PixelFormat pixelFormat, const IntSize& size)
{
    return mustFitInInt32(computeRawPixelComponentCount(pixelFormat, size));
}

CheckedUint32 PixelBuffer::computeBufferSize(PixelFormat pixelFormat, const IntSize& size)
{
    // FIXME: Implement a better way to deal with sizes of diffferent formats.
    unsigned bytesPerPixelComponent =
#if HAVE(HDR_SUPPORT)
        (pixelFormat == PixelFormat::RGBA16F) ? 2 :
#endif
        1;
    return mustFitInInt32(computeRawPixelComponentCount(pixelFormat, size) * bytesPerPixelComponent);
}

PixelBuffer::PixelBuffer(const PixelBufferFormat& format, const IntSize& size, std::span<uint8_t> bytes)
    : m_format(format)
    , m_size(size)
    , m_bytes(bytes)
{
    RELEASE_ASSERT_WITH_SECURITY_IMPLICATION((m_size.area() * 4) <= m_bytes.size());
}

PixelBuffer::~PixelBuffer() = default;

bool PixelBuffer::setRange(std::span<const uint8_t> data, size_t byteOffset)
{
    if (!isSumSmallerThanOrEqual(byteOffset, data.size(), m_bytes.size()))
        return false;

    memmoveSpan(m_bytes.subspan(byteOffset), data);
    return true;
}

bool PixelBuffer::zeroRange(size_t byteOffset, size_t rangeByteLength)
{
    if (!isSumSmallerThanOrEqual(byteOffset, rangeByteLength, m_bytes.size()))
        return false;

    zeroSpan(m_bytes.subspan(byteOffset, rangeByteLength));
    return true;
}

uint8_t PixelBuffer::item(size_t index) const
{
    RELEASE_ASSERT_WITH_SECURITY_IMPLICATION(index < m_bytes.size());
    return m_bytes[index];
}

void PixelBuffer::set(size_t index, double value)
{
    RELEASE_ASSERT_WITH_SECURITY_IMPLICATION(index < m_bytes.size());
    m_bytes[index] = JSC::Uint8ClampedAdaptor::toNativeFromDouble(value);
}

} // namespace WebCore
