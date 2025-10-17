/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 21, 2023.
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
#include "DMABufBuffer.h"

#if USE(GBM)
#include "CoordinatedPlatformLayerBuffer.h"
#include <atomic>

namespace WebCore {

static uint64_t generateID()
{
    static std::atomic<uint64_t> id;
    return ++id;
}

DMABufBuffer::DMABufBuffer(const IntSize& size, uint32_t fourcc, Vector<WTF::UnixFileDescriptor>&& fds, Vector<uint32_t>&& offsets, Vector<uint32_t>&& strides, uint64_t modifier)
    : m_id(generateID())
    , m_attributes({ size, fourcc, WTFMove(fds), WTFMove(offsets), WTFMove(strides), modifier })
{
}

DMABufBuffer::DMABufBuffer(uint64_t id, Attributes&& attributes)
    : m_id(id)
    , m_attributes(WTFMove(attributes))
{
}

DMABufBuffer::~DMABufBuffer() = default;

void DMABufBuffer::setBuffer(std::unique_ptr<CoordinatedPlatformLayerBuffer>&& buffer)
{
    m_buffer = WTFMove(buffer);
}

std::optional<DMABufBuffer::Attributes> DMABufBuffer::takeAttributes()
{
    if (m_attributes.fds.isEmpty())
        return std::nullopt;

    return DMABufBuffer::Attributes { WTFMove(m_attributes.size), std::exchange(m_attributes.fourcc, 0), WTFMove(m_attributes.fds), WTFMove(m_attributes.offsets), WTFMove(m_attributes.strides), std::exchange(m_attributes.modifier, 0) };
}

} // namespace WebCore

#endif // USE(GBM)
