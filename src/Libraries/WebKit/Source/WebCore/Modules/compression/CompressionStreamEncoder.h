/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 13, 2023.
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
#include "ZStream.h"
#include <JavaScriptCore/Forward.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/Vector.h>

namespace WebCore {

class CompressionStreamEncoder : public RefCounted<CompressionStreamEncoder> {
public:
    static ExceptionOr<Ref<CompressionStreamEncoder>> create(unsigned char formatChar)
    {
        auto format = static_cast<Formats::CompressionFormat>(formatChar);
#if !PLATFORM(COCOA)
        if (format == Formats::CompressionFormat::Brotli)
            return Exception { ExceptionCode::NotSupportedError, "Unsupported algorithm"_s };
#endif
        return adoptRef(*new CompressionStreamEncoder(format));
    }

    ExceptionOr<RefPtr<Uint8Array>> encode(const BufferSource&&);
    ExceptionOr<RefPtr<Uint8Array>> flush();

private:
    bool didDeflateFinish(int) const;

    ExceptionOr<Ref<JSC::ArrayBuffer>> compress(std::span<const uint8_t>);
    ExceptionOr<Ref<JSC::ArrayBuffer>> compressZlib(std::span<const uint8_t>);
#if PLATFORM(COCOA)
    bool didDeflateFinishAppleCompressionFramework(int);
    ExceptionOr<Ref<JSC::ArrayBuffer>> compressAppleCompressionFramework(std::span<const uint8_t>);
#endif

    explicit CompressionStreamEncoder(Formats::CompressionFormat format)
        : m_format(format)
    {
    }

    // If the user provides too small of an input size we will automatically allocate a page worth of memory instead.
    // Very small input sizes can result in a larger output than their input. This would require an additional 
    // encode call then, which is not desired.
    const size_t startingAllocationSize = 16384; // 16KB
    const size_t maxAllocationSize = 1073741824; // 1GB

    bool m_didFinish { false };
    const Formats::CompressionFormat m_format;

    // TODO: convert to using variant
    CompressionStream m_compressionStream;
    ZStream m_zstream;
};
} // namespace WebCore
