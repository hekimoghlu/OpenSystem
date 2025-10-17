/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 30, 2024.
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
#include "ByteArrayPixelBuffer.h"

#include <JavaScriptCore/TypedArrayInlines.h>

namespace WebCore {

Ref<ByteArrayPixelBuffer> ByteArrayPixelBuffer::create(const PixelBufferFormat& format, const IntSize& size, JSC::Uint8ClampedArray& data)
{
    ASSERT(format.pixelFormat == PixelFormat::RGBA8 || format.pixelFormat == PixelFormat::BGRA8);
    return adoptRef(*new ByteArrayPixelBuffer(format, size, { data }));
}

std::optional<Ref<ByteArrayPixelBuffer>> ByteArrayPixelBuffer::create(const PixelBufferFormat& format, const IntSize& size, std::span<const uint8_t> data)
{
    if (!(format.pixelFormat == PixelFormat::RGBA8 || format.pixelFormat == PixelFormat::BGRA8)) {
        ASSERT_NOT_REACHED();
        return std::nullopt;
    }

    auto computedBufferSize = PixelBuffer::computeBufferSize(format.pixelFormat, size);
    if (computedBufferSize.hasOverflowed()) {
        ASSERT_NOT_REACHED();
        return std::nullopt;
    }

    if (data.size_bytes() != computedBufferSize.value()) {
        ASSERT_NOT_REACHED();
        return std::nullopt;
    }

    auto buffer = Uint8ClampedArray::tryCreate(data.data(), data.size_bytes());
    if (!buffer) {
        ASSERT_NOT_REACHED();
        return std::nullopt;
    }

    return ByteArrayPixelBuffer::create(format, size, buffer.releaseNonNull());
}

RefPtr<ByteArrayPixelBuffer> ByteArrayPixelBuffer::tryCreate(const PixelBufferFormat& format, const IntSize& size)
{
    ASSERT(supportedPixelFormat(format.pixelFormat));

    if (!(format.pixelFormat == PixelFormat::RGBA8 || format.pixelFormat == PixelFormat::BGRA8)) {
        ASSERT_NOT_REACHED();
        return nullptr;
    }

    auto bufferSize = computeBufferSize(format.pixelFormat, size);
    if (bufferSize.hasOverflowed())
        return nullptr;

    auto data = Uint8ClampedArray::tryCreateUninitialized(bufferSize);
    if (!data)
        return nullptr;

    return create(format, size, data.releaseNonNull());
}

RefPtr<ByteArrayPixelBuffer> ByteArrayPixelBuffer::tryCreate(const PixelBufferFormat& format, const IntSize& size, Ref<JSC::ArrayBuffer>&& arrayBuffer)
{
    ASSERT(supportedPixelFormat(format.pixelFormat));

    if (!(format.pixelFormat == PixelFormat::RGBA8 || format.pixelFormat == PixelFormat::BGRA8)) {
        ASSERT_NOT_REACHED();
        return nullptr;
    }

    auto bufferSize = computeBufferSize(format.pixelFormat, size);
    if (bufferSize.hasOverflowed())
        return nullptr;
    if (bufferSize != arrayBuffer->byteLength())
        return nullptr;

    Ref data = Uint8ClampedArray::create(WTFMove(arrayBuffer));
    return create(format, size, WTFMove(data));
}

ByteArrayPixelBuffer::ByteArrayPixelBuffer(const PixelBufferFormat& format, const IntSize& size, Ref<JSC::Uint8ClampedArray>&& data)
    : PixelBuffer(format, size, data->mutableSpan())
    , m_data(WTFMove(data))
{
}

RefPtr<PixelBuffer> ByteArrayPixelBuffer::createScratchPixelBuffer(const IntSize& size) const
{
    return ByteArrayPixelBuffer::tryCreate(m_format, size);
}

std::span<const uint8_t> ByteArrayPixelBuffer::span() const
{
    Ref data = m_data;
    ASSERT(data->byteLength() == (m_size.area() * 4));
    return data->span();
}

} // namespace WebCore
