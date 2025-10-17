/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 28, 2025.
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
#include <compression.h>

namespace WebCore {

// The compression algorithm is broken up into 2 steps.
// 1. Compression of Data
// 2. Flush Remaining Data
//
// When src_size is empty we can normally exit performing compression, but during the flush
// step we may have data buffered and will need to continue to keep flushing out the rest.
bool CompressionStreamEncoder::didDeflateFinishAppleCompressionFramework(int result)
{
    return !m_compressionStream.getPlatformStream().src_size && (!m_didFinish || (m_didFinish && result == COMPRESSION_STATUS_END));
}

ExceptionOr<Ref<JSC::ArrayBuffer>> CompressionStreamEncoder::compressAppleCompressionFramework(std::span<const uint8_t> input)
{
    size_t allocateSize = std::max(input.size(), startingAllocationSize);
    auto storage = SharedBufferBuilder();

    compression_status result;
    bool shouldDecompress = true;

    if (!m_compressionStream.initializeIfNecessary(CompressionStream::Algorithm::Brotli, CompressionStream::Operation::Compression))
        return Exception { ExceptionCode::TypeError, "Initialization Failed."_s };

    m_compressionStream.getPlatformStream().src_ptr = input.data();
    m_compressionStream.getPlatformStream().src_size = input.size();

    while (shouldDecompress) {
        Vector<uint8_t> output;
        if (!output.tryReserveInitialCapacity(allocateSize)) {
            allocateSize /= 4;

            if (allocateSize < startingAllocationSize)
                return Exception { ExceptionCode::OutOfMemoryError };

            continue;
        }

        output.grow(allocateSize);

        m_compressionStream.getPlatformStream().dst_ptr = output.data();
        m_compressionStream.getPlatformStream().dst_size = output.size();

        result = compression_stream_process(&m_compressionStream.getPlatformStream(), m_didFinish ? COMPRESSION_STREAM_FINALIZE : 0);

        if (result == COMPRESSION_STATUS_ERROR)
            return Exception { ExceptionCode::TypeError, "Failed to Encode Data."_s };

        if ((result == COMPRESSION_STATUS_END && m_compressionStream.getPlatformStream().src_size)
            || (m_didFinish && m_compressionStream.getPlatformStream().src_size))
            return Exception { ExceptionCode::TypeError, "Extra bytes past the end."_s };

        if (didDeflateFinishAppleCompressionFramework(result)) {
            shouldDecompress = false;
            output.shrink(allocateSize - m_compressionStream.getPlatformStream().dst_size);
        } else {
            if (allocateSize < maxAllocationSize)
                allocateSize *= 2;
        }

        storage.append(output);
    }

    RefPtr decompressedData = storage.takeAsArrayBuffer();
    if (!decompressedData)
        return Exception { ExceptionCode::OutOfMemoryError };

    return decompressedData.releaseNonNull();
}

}
