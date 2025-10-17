/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 1, 2023.
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

#include "IPCSemaphore.h"
#include "StreamConnectionBuffer.h"
#include "StreamConnectionEncoder.h"

namespace IPC {

class StreamServerConnectionBuffer : public StreamConnectionBuffer {
public:
    static std::optional<StreamServerConnectionBuffer> map(Handle&&);
    StreamServerConnectionBuffer(StreamServerConnectionBuffer&&) = default;
    StreamServerConnectionBuffer& operator=(StreamServerConnectionBuffer&&) = default;
    std::optional<std::span<uint8_t>> tryAcquire();
    std::span<uint8_t> acquireAll();
    enum class WakeUpClient : bool { No, Yes };
    WakeUpClient release(size_t readSize);
    WakeUpClient releaseAll();

private:
    using StreamConnectionBuffer::StreamConnectionBuffer;
    static constexpr size_t minimumMessageSize = StreamConnectionEncoder::minimumMessageSize;
    static constexpr size_t messageAlignment = StreamConnectionEncoder::messageAlignment;
    std::span<uint8_t> alignedMutableSpan(size_t offset, size_t limit);
    size_t size(size_t offset, size_t limit) const;
    size_t alignOffset(size_t offset) const { return StreamConnectionBuffer::alignOffset<messageAlignment>(offset, minimumMessageSize); }
    using ServerLimit = ClientOffset;
    Atomic<ServerLimit>& sharedServerLimit() { return clientOffset(); }
    Atomic<ServerOffset>& sharedServerOffset() { return serverOffset(); }
    size_t clampedLimit(ServerLimit) const;

    size_t m_serverOffset { 0 };
};

inline std::optional<StreamServerConnectionBuffer> StreamServerConnectionBuffer::map(Handle&& handle)
{
    auto sharedMemory = WebCore::SharedMemory::map(WTFMove(handle), WebCore::SharedMemory::Protection::ReadWrite);
    if (UNLIKELY(!sharedMemory))
        return std::nullopt;
    return StreamServerConnectionBuffer { sharedMemory.releaseNonNull() };
}

inline std::optional<std::span<uint8_t>> StreamServerConnectionBuffer::tryAcquire()
{
    ServerLimit serverLimit = sharedServerLimit().load(std::memory_order_acquire);
    if (serverLimit == ServerLimit::serverIsSleepingTag)
        return std::nullopt;

    auto result = alignedMutableSpan(m_serverOffset, clampedLimit(serverLimit));
    if (result.size() < minimumMessageSize) {
        serverLimit = sharedServerLimit().compareExchangeStrong(serverLimit, ServerLimit::serverIsSleepingTag, std::memory_order_acq_rel, std::memory_order_acquire);
        result = alignedMutableSpan(m_serverOffset, clampedLimit(serverLimit));
    }

    if (result.size() < minimumMessageSize)
        return std::nullopt;

    return result;
}

inline std::span<uint8_t> StreamServerConnectionBuffer::acquireAll()
{
    return alignedMutableSpan(0, dataSize() - 1);
}

inline StreamServerConnectionBuffer::WakeUpClient StreamServerConnectionBuffer::release(size_t readSize)
{
    ASSERT(readSize);
    readSize = std::max(readSize, minimumMessageSize);
    ServerOffset serverOffset = static_cast<ServerOffset>(wrapOffset(alignOffset(m_serverOffset) + readSize));

    ServerOffset oldServerOffset = sharedServerOffset().exchange(serverOffset, std::memory_order_acq_rel);
    WakeUpClient wakeUpClient = WakeUpClient::No;
    // If the client wrote over serverOffset, it means the client is waiting.
    if (oldServerOffset == ServerOffset::clientIsWaitingTag)
        wakeUpClient = WakeUpClient::Yes;
    else
        ASSERT(!(oldServerOffset & ServerOffset::clientIsWaitingTag));

    m_serverOffset = serverOffset;
    return wakeUpClient;
}

inline StreamServerConnectionBuffer::WakeUpClient StreamServerConnectionBuffer::releaseAll()
{
    sharedServerLimit().store(static_cast<ServerLimit>(0), std::memory_order_release);
    ServerOffset oldServerOffset = sharedServerOffset().exchange(static_cast<ServerOffset>(0), std::memory_order_acq_rel);
    WakeUpClient wakeUpClient = WakeUpClient::No;
    // If the client wrote over serverOffset, it means the client is waiting.
    if (oldServerOffset == ServerOffset::clientIsWaitingTag)
        wakeUpClient = WakeUpClient::Yes;
    else
        ASSERT(!(oldServerOffset & ServerOffset::clientIsWaitingTag));
    m_serverOffset = 0;
    return wakeUpClient;
}

inline std::span<uint8_t> StreamServerConnectionBuffer::alignedMutableSpan(size_t offset, size_t limit)
{
    ASSERT(offset < dataSize());
    ASSERT(limit < dataSize());
    size_t aligned = alignOffset(offset);
    size_t resultSize = 0;
    if (offset < limit) {
        if (offset <= aligned && aligned < limit)
            resultSize = size(aligned, limit);
    } else if (offset > limit) {
        if (aligned >= offset || aligned < limit)
            resultSize = size(aligned, limit);
    }
    return mutableSpan().subspan(aligned, resultSize);
}

inline size_t StreamServerConnectionBuffer::size(size_t offset, size_t limit) const
{
    if (offset <= limit)
        return limit - offset;
    return dataSize() - offset;
}

inline size_t StreamServerConnectionBuffer::clampedLimit(ServerLimit serverLimit) const
{
    ASSERT(!(serverLimit & ServerLimit::serverIsSleepingTag));
    size_t limit = static_cast<size_t>(serverLimit);
    ASSERT(limit <= dataSize() - 1);
    return std::min(limit, dataSize() - 1);
}

}
