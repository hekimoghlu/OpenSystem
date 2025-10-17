/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 28, 2023.
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

#if USE(COORDINATED_GRAPHICS) && USE(GBM)
#include "IntSize.h"
#include <optional>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/Vector.h>
#include <wtf/unix/UnixFileDescriptor.h>

namespace WebCore {

class CoordinatedPlatformLayerBuffer;

struct DMABufBufferAttributes {
    IntSize size;
    uint32_t fourcc { 0 };
    Vector<WTF::UnixFileDescriptor> fds;
    Vector<uint32_t> offsets;
    Vector<uint32_t> strides;
    uint64_t modifier { 0 };
};

class DMABufBuffer final : public ThreadSafeRefCounted<DMABufBuffer> {
public:
    using Attributes = DMABufBufferAttributes;

    static Ref<DMABufBuffer> create(const IntSize& size, uint32_t fourcc, Vector<WTF::UnixFileDescriptor>&& fds, Vector<uint32_t>&& offsets, Vector<uint32_t>&& strides, uint64_t modifier)
    {
        return adoptRef(*new DMABufBuffer(size, fourcc, WTFMove(fds), WTFMove(offsets), WTFMove(strides), modifier));
    }
    static Ref<DMABufBuffer> create(uint64_t id, Attributes&& attributes)
    {
        return adoptRef(*new DMABufBuffer(id, WTFMove(attributes)));
    }
    ~DMABufBuffer();

    uint64_t id() const { return m_id; }
    const Attributes& attributes() const { return m_attributes; }
    std::optional<Attributes> takeAttributes();

    enum class ColorSpace : uint8_t { BT601, BT709, BT2020, SMPTE240M };
    std::optional<ColorSpace> colorSpace() const { return m_colorSpace; }
    void setColorSpace(ColorSpace colorSpace) { m_colorSpace = colorSpace; }

    CoordinatedPlatformLayerBuffer* buffer() const { return m_buffer.get(); }
    void setBuffer(std::unique_ptr<CoordinatedPlatformLayerBuffer>&&);

private:
    DMABufBuffer(const IntSize&, uint32_t fourcc, Vector<WTF::UnixFileDescriptor>&&, Vector<uint32_t>&&, Vector<uint32_t>&&, uint64_t modifier);
    DMABufBuffer(uint64_t id, Attributes&&);

    uint64_t m_id { 0 };
    Attributes m_attributes;
    std::optional<ColorSpace> m_colorSpace;
    std::unique_ptr<CoordinatedPlatformLayerBuffer> m_buffer;
};

} // namespace WebCore

#endif // USE(COORDINATED_GRAPHICS) && USE(GBM)
