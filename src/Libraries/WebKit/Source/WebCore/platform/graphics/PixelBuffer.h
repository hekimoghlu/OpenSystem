/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 27, 2024.
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

#include "IntSize.h"
#include "PixelBufferFormat.h"
#include <wtf/RefCounted.h>

namespace WTF {
class TextStream;
}

namespace WebCore {

class PixelBuffer : public RefCounted<PixelBuffer> {
    WTF_MAKE_NONCOPYABLE(PixelBuffer);
public:
    static CheckedUint32 computePixelCount(const IntSize&);
    static CheckedUint32 computePixelComponentCount(PixelFormat, const IntSize&);
    WEBCORE_EXPORT static CheckedUint32 computeBufferSize(PixelFormat, const IntSize&);

    WEBCORE_EXPORT static bool supportedPixelFormat(PixelFormat);

    WEBCORE_EXPORT virtual ~PixelBuffer();

    const PixelBufferFormat& format() const { return m_format; }
    const IntSize& size() const { return m_size; }

    std::span<uint8_t> bytes() const { return m_bytes; }

    enum class Type {
        ByteArray,
#if HAVE(HDR_SUPPORT)
        Float16Array,
#endif
        Other
    };
    virtual Type type() const { return Type::Other; }
    virtual RefPtr<PixelBuffer> createScratchPixelBuffer(const IntSize&) const = 0;

    bool setRange(std::span<const uint8_t> data, size_t byteOffset);
    WEBCORE_EXPORT bool zeroRange(size_t byteOffset, size_t rangeByteLength);
    void zeroFill() { zeroRange(0, bytes().size()); }

    WEBCORE_EXPORT uint8_t item(size_t index) const;
    void set(size_t index, double value);

protected:
    WEBCORE_EXPORT PixelBuffer(const PixelBufferFormat&, const IntSize&, std::span<uint8_t> bytes);
    
    PixelBufferFormat m_format;
    IntSize m_size;

    std::span<uint8_t> m_bytes;
};

} // namespace WebCore
