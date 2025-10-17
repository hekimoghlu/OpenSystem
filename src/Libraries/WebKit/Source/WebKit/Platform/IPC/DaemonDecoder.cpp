/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 7, 2022.
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
#include "DaemonDecoder.h"

#include <wtf/StdLibExtras.h>

namespace WebKit {

namespace Daemon {

Decoder::~Decoder()
{
    ASSERT(m_bufferPosition == m_buffer.size());
}

bool Decoder::bufferIsLargeEnoughToContainBytes(size_t bytes) const
{
    return bytes <= m_buffer.size() - m_bufferPosition;
}

bool Decoder::decodeFixedLengthData(std::span<uint8_t> data)
{
    if (!bufferIsLargeEnoughToContainBytes(data.size()))
        return false;
    memcpySpan(data, m_buffer.subspan(m_bufferPosition, data.size()));
    m_bufferPosition += data.size();
    return true;
}

std::span<const uint8_t> Decoder::decodeFixedLengthReference(size_t size)
{
    if (!bufferIsLargeEnoughToContainBytes(size))
        return { };
    auto data = m_buffer.subspan(m_bufferPosition, size);
    m_bufferPosition += size;
    return data;
}

} // namespace Daemon

} // namespace WebKit
