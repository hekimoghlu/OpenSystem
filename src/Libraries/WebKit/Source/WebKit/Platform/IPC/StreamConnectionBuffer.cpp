/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 1, 2022.
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
#include "StreamConnectionBuffer.h"

#include "Decoder.h"

namespace IPC {

StreamConnectionBuffer::StreamConnectionBuffer(Ref<WebCore::SharedMemory>&& memory)
    : m_dataSize(memory->size() - headerSize())
    , m_sharedMemory(WTFMove(memory))
{
    ASSERT(sharedMemorySizeIsValid(m_sharedMemory->size()));
}

StreamConnectionBuffer::~StreamConnectionBuffer() = default;

StreamConnectionBuffer::Handle StreamConnectionBuffer::createHandle()
{
    auto handle = Ref { m_sharedMemory }->createHandle(WebCore::SharedMemory::Protection::ReadWrite);
    if (!handle)
        CRASH();
    return { WTFMove(*handle) };
}

std::span<uint8_t> StreamConnectionBuffer::headerForTesting()
{
    return m_sharedMemory->mutableSpan().first(headerSize());
}

}
