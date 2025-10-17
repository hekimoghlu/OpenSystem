/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 25, 2024.
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
#include "Gunzip.h"

#include <compression.h>

namespace PAL {

Vector<LChar> gunzip(std::span<const uint8_t> data)
{
    Vector<LChar> result;

    // Parse the gzip header.
    auto checks = [&]() {
        return data.size() >= 10 && data[0] == 0x1f && data[1] == 0x8b && data[2] == 0x8 && data[3] == 0x0;
    };
    ASSERT(checks());
    if (!checks())
        return { };

    constexpr auto ignoredByteCount = 10;

    compression_stream stream;
    auto status = compression_stream_init(&stream, COMPRESSION_STREAM_DECODE, COMPRESSION_ZLIB);
    ASSERT(status == COMPRESSION_STATUS_OK);
    if (status != COMPRESSION_STATUS_OK)
        return { };
    stream.dst_ptr = result.data();
    stream.dst_size = result.size();
    stream.src_ptr = data.subspan(ignoredByteCount).data();
    stream.src_size = data.size() - ignoredByteCount;
    size_t offset = 0;

    do {
        uint8_t* originalDestinationPointer = stream.dst_ptr;
        status = compression_stream_process(&stream, COMPRESSION_STREAM_FINALIZE);
        uint8_t* newDestinationPointer = stream.dst_ptr;
        offset += newDestinationPointer - originalDestinationPointer;
        switch (status) {
        case COMPRESSION_STATUS_OK: {
            auto newSize = offset * 1.5 + 1;
            // FIXME: We can get better performance if we, instead of resizing the buffer, we allocate distinct chunks and have a postprocessing step which concatenates the chunks together.
            if (newSize > result.size())
                result.grow(newSize);
            stream.dst_ptr = result.mutableSpan().subspan(offset).data();
            stream.dst_size = result.size() - offset;
            break;
        }
        case COMPRESSION_STATUS_END:
            status = compression_stream_destroy(&stream);
            ASSERT(status == COMPRESSION_STATUS_OK);
            if (status == COMPRESSION_STATUS_OK) {
                result.shrink(stream.dst_ptr - result.data());
                return result;
            }
            return { };
        case COMPRESSION_STATUS_ERROR:
        default:
            ASSERT_NOT_REACHED();
            compression_stream_destroy(&stream);
            return { };
        }
    } while (true);
}

} // namespace WTF
