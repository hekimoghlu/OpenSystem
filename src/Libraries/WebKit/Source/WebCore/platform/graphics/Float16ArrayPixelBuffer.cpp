/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 20, 2021.
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
#include "Float16ArrayPixelBuffer.h"

#if HAVE(HDR_SUPPORT)

#include <JavaScriptCore/TypedArrayInlines.h>

namespace WebCore {

Ref<Float16ArrayPixelBuffer> Float16ArrayPixelBuffer::create(const PixelBufferFormat& format, const IntSize& size, JSC::Float16Array& data)
{
    ASSERT(format.pixelFormat == PixelFormat::RGBA16F);
    return adoptRef(*new Float16ArrayPixelBuffer(format, size, { data }));
}

std::optional<Ref<Float16ArrayPixelBuffer>> Float16ArrayPixelBuffer::create(const PixelBufferFormat& format, const IntSize& size, std::span<const Float16> data)
{
    if (format.pixelFormat != PixelFormat::RGBA16F) {
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

    auto buffer = JSC::Float16Array::tryCreate(data);
    if (!buffer) {
        ASSERT_NOT_REACHED();
        return std::nullopt;
    }

    return Float16ArrayPixelBuffer::create(format, size, buffer.releaseNonNull());
}

RefPtr<Float16ArrayPixelBuffer> Float16ArrayPixelBuffer::tryCreate(const PixelBufferFormat& format, const IntSize& size)
{
    ASSERT(supportedPixelFormat(format.pixelFormat));

    if (format.pixelFormat != PixelFormat::RGBA16F) {
        ASSERT_NOT_REACHED();
        return nullptr;
    }

    auto bufferSize = computeBufferSize(format.pixelFormat, size);
    if (bufferSize.hasOverflowed())
        return nullptr;

    auto data = JSC::Float16Array::tryCreateUninitialized(bufferSize / sizeof(Float16));
    if (!data)
        return nullptr;

    return create(format, size, data.releaseNonNull());
}

RefPtr<Float16ArrayPixelBuffer> Float16ArrayPixelBuffer::tryCreate(const PixelBufferFormat& format, const IntSize& size, Ref<JSC::ArrayBuffer>&& arrayBuffer)
{
    ASSERT(supportedPixelFormat(format.pixelFormat));

    if (format.pixelFormat != PixelFormat::RGBA16F) {
        ASSERT_NOT_REACHED();
        return nullptr;
    }

    auto bufferSize = computeBufferSize(format.pixelFormat, size);
    if (bufferSize.hasOverflowed())
        return nullptr;
    if (bufferSize != arrayBuffer->byteLength())
        return nullptr;

    Ref data = JSC::Float16Array::create(WTFMove(arrayBuffer));
    return create(format, size, WTFMove(data));
}

Float16ArrayPixelBuffer::Float16ArrayPixelBuffer(const PixelBufferFormat& format, const IntSize& size, Ref<JSC::Float16Array>&& data)
    : PixelBuffer(format, size, data->mutableSpan())
    , m_data(WTFMove(data))
{
}

RefPtr<PixelBuffer> Float16ArrayPixelBuffer::createScratchPixelBuffer(const IntSize& size) const
{
    return Float16ArrayPixelBuffer::tryCreate(m_format, size);
}

std::span<const uint8_t> Float16ArrayPixelBuffer::span() const
{
    Ref data = m_data;
    ASSERT(data->byteLength() == (m_size.area() * 4 * sizeof(Float16)));
    return data->span();
}

} // namespace WebCore

#endif // HAVE(HDR_SUPPORT)
