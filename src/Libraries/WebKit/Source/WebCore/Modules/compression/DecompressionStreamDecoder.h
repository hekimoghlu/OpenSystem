/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 16, 2024.
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

#include "BufferSource.h"
#include "CompressionStream.h"
#include "ExceptionOr.h"
#include "Formats.h"
#include "SharedBuffer.h"
#include "ZStream.h"
#include <JavaScriptCore/ArrayBuffer.h>
#include <JavaScriptCore/Forward.h>
#include <cstring>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <zlib.h>

namespace WebCore {

class DecompressionStreamDecoder : public RefCounted<DecompressionStreamDecoder> {
public:
    static ExceptionOr<Ref<DecompressionStreamDecoder>> create(unsigned char formatChar)
    {
        auto format = static_cast<Formats::CompressionFormat>(formatChar);
#if !PLATFORM(COCOA)
        if (format == Formats::CompressionFormat::Brotli)
            return Exception { ExceptionCode::NotSupportedError, "Unsupported algorithm"_s };
#endif
        return adoptRef(*new DecompressionStreamDecoder(format));
    }

    ExceptionOr<RefPtr<Uint8Array>> decode(const BufferSource&&);
    ExceptionOr<RefPtr<Uint8Array>> flush();

private:
    bool didInflateFinish(int) const;
    bool didInflateContainExtraBytes(int) const;

    ExceptionOr<Ref<JSC::ArrayBuffer>> decompress(std::span<const uint8_t>);
    ExceptionOr<Ref<JSC::ArrayBuffer>> decompressZlib(std::span<const uint8_t>);
#if PLATFORM(COCOA)
    bool didInflateFinishAppleCompressionFramework(int);
    ExceptionOr<Ref<JSC::ArrayBuffer>> decompressAppleCompressionFramework(std::span<const uint8_t>);
#endif

    explicit DecompressionStreamDecoder(Formats::CompressionFormat format)
        : m_format(format)
    {
    }

    // When given an encoded input, it is difficult to guess the output size.
    // My approach here is starting from one page and growing at a linear rate of x2 until the input data
    // has been fully processed. To ensure the user's memory is not completely consumed, I am setting a cap
    // of 1GB per allocation. This strategy enables very fast memory allocation growth without needing to perform
    // unnecessarily large allocations upfront.
    const size_t startingAllocationSize = 16384; // 16KB
    const size_t maxAllocationSize = 1073741824; // 1GB

    bool m_didFinish { false };
    const Formats::CompressionFormat m_format;

    // TODO: convert to using variant
    CompressionStream m_compressionStream;
    ZStream m_zstream;
};
} // namespace WebCore
