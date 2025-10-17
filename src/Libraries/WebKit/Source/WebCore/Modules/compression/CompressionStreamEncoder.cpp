/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 26, 2025.
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
#include "CompressionStreamEncoder.h"

#include "BufferSource.h"
#include "Exception.h"
#include "SharedBuffer.h"
#include <JavaScriptCore/GenericTypedArrayViewInlines.h>
#include <JavaScriptCore/JSCInlines.h>
#include <JavaScriptCore/JSGenericTypedArrayViewInlines.h>

namespace WebCore {

ExceptionOr<RefPtr<Uint8Array>> CompressionStreamEncoder::encode(const BufferSource&& input)
{
    auto compressedDataCheck = compress(input.span());
    if (compressedDataCheck.hasException())
        return compressedDataCheck.releaseException();

    Ref compressedData = compressedDataCheck.releaseReturnValue();
    if (!compressedData->byteLength())
        return nullptr;

    return RefPtr { Uint8Array::create(WTFMove(compressedData)) };
}

ExceptionOr<RefPtr<Uint8Array>> CompressionStreamEncoder::flush()
{
    m_didFinish = true;

    auto compressedDataCheck = compress({ });
    if (compressedDataCheck.hasException())
        return compressedDataCheck.releaseException();

    Ref compressedData = compressedDataCheck.releaseReturnValue();
    if (!compressedData->byteLength())
        return nullptr;

    return RefPtr { Uint8Array::create(WTFMove(compressedData)) };
}

// The compression algorithm is broken up into 2 steps.
// 1. Compression of Data
// 2. Flush Remaining Data
//
// When avail_in is empty we can normally exit performing compression, but during the flush
// step we may have data buffered and will need to continue to keep flushing out the rest.
bool CompressionStreamEncoder::didDeflateFinish(int result) const
{
    return !m_zstream.getPlatformStream().avail_in && (!m_didFinish || (m_didFinish && result == Z_STREAM_END));
}

// See https://www.zlib.net/manual.html#Constants
static bool didDeflateFail(int result)
{
    return result != Z_OK && result != Z_STREAM_END && result != Z_BUF_ERROR;
}

ExceptionOr<Ref<JSC::ArrayBuffer>> CompressionStreamEncoder::compress(std::span<const uint8_t> input)
{
#if PLATFORM(COCOA)
    if (m_format == Formats::CompressionFormat::Brotli)
        return compressAppleCompressionFramework(input);
#endif
    return compressZlib(input);
}

static ZStream::Algorithm compressionAlgorithm(Formats::CompressionFormat format)
{
    switch (format) {
    case Formats::CompressionFormat::Brotli:
        RELEASE_ASSERT_NOT_REACHED();
    case Formats::CompressionFormat::Gzip:
        return ZStream::Algorithm::Gzip;
    case Formats::CompressionFormat::Zlib:
        return ZStream::Algorithm::Zlib;
    case Formats::CompressionFormat::Deflate:
        return ZStream::Algorithm::Deflate;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

ExceptionOr<Ref<JSC::ArrayBuffer>> CompressionStreamEncoder::compressZlib(std::span<const uint8_t> input)
{
    size_t allocateSize = std::max(input.size(), startingAllocationSize);
    auto storage = SharedBufferBuilder();

    int result;    
    bool shouldCompress = true;

    if (!m_zstream.initializeIfNecessary(compressionAlgorithm(m_format), ZStream::Operation::Compression))
        return Exception { ExceptionCode::TypeError, "Initialization Failed."_s };

    m_zstream.getPlatformStream().next_in = const_cast<z_const Bytef*>(input.data());
    m_zstream.getPlatformStream().avail_in = input.size();

    while (shouldCompress) {
        Vector<uint8_t> output;
        if (!output.tryReserveInitialCapacity(allocateSize)) {
            allocateSize /= 4;

            if (allocateSize < startingAllocationSize)
                return Exception { ExceptionCode::OutOfMemoryError };

            continue;
        }

        output.grow(allocateSize);

        m_zstream.getPlatformStream().next_out = output.data();
        m_zstream.getPlatformStream().avail_out = output.size();

        result = deflate(&m_zstream.getPlatformStream(), m_didFinish ? Z_FINISH : Z_NO_FLUSH);

        if (didDeflateFail(result))
            return Exception { ExceptionCode::TypeError, "Failed to compress data."_s };

        if (didDeflateFinish(result)) {
            shouldCompress = false;
            output.shrink(allocateSize - m_zstream.getPlatformStream().avail_out);
        }
        else {
            if (allocateSize < maxAllocationSize)
                allocateSize *= 2;
        }

        storage.append(output);
    }

    RefPtr compressedData = storage.takeAsArrayBuffer();
    if (!compressedData)
        return Exception { ExceptionCode::OutOfMemoryError };

    return compressedData.releaseNonNull();
}

} // namespace WebCore
