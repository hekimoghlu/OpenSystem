/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 21, 2024.
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
#include "Encoder.h"

#include "ArgumentCoders.h"
#include "MessageFlags.h"
#include <algorithm>
#include <wtf/OptionSet.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/UniqueRef.h>

#if OS(DARWIN)
#include <wtf/Mmap.h>
#endif

namespace IPC {

static constexpr uint8_t defaultMessageFlags = 0;

#if OS(DARWIN)
static inline MallocSpan<uint8_t, Mmap> allocateBuffer(size_t size)
{
    auto buffer = MallocSpan<uint8_t, Mmap>::mmap(size, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1);
    RELEASE_ASSERT(!!buffer);
    return buffer;
}
#else
static inline MallocSpan<uint8_t> allocateBuffer(size_t size)
{
    return MallocSpan<uint8_t>::malloc(size);
}
#endif

WTF_MAKE_TZONE_ALLOCATED_IMPL(Encoder);

Encoder::Encoder(MessageName messageName, uint64_t destinationID)
    : m_messageName(messageName)
    , m_destinationID(destinationID)
{
    encodeHeader();
}

// FIXME: We need to dispose of the attachments in cases of failure.
Encoder::~Encoder() = default;

void Encoder::setShouldDispatchMessageWhenWaitingForSyncReply(ShouldDispatchWhenWaitingForSyncReply shouldDispatchWhenWaitingForSyncReply)
{
    switch (shouldDispatchWhenWaitingForSyncReply) {
    case ShouldDispatchWhenWaitingForSyncReply::No:
        messageFlags().remove(MessageFlags::DispatchMessageWhenWaitingForSyncReply);
        messageFlags().remove(MessageFlags::DispatchMessageWhenWaitingForUnboundedSyncReply);
        break;
    case ShouldDispatchWhenWaitingForSyncReply::Yes:
        messageFlags().add(MessageFlags::DispatchMessageWhenWaitingForSyncReply);
        messageFlags().remove(MessageFlags::DispatchMessageWhenWaitingForUnboundedSyncReply);
        break;
    case ShouldDispatchWhenWaitingForSyncReply::YesDuringUnboundedIPC:
        messageFlags().remove(MessageFlags::DispatchMessageWhenWaitingForSyncReply);
        messageFlags().add(MessageFlags::DispatchMessageWhenWaitingForUnboundedSyncReply);
        break;
    }
}

bool Encoder::isFullySynchronousModeForTesting() const
{
    return messageFlags().contains(MessageFlags::UseFullySynchronousModeForTesting);
}

void Encoder::setFullySynchronousModeForTesting()
{
    messageFlags().add(MessageFlags::UseFullySynchronousModeForTesting);
}

void Encoder::setShouldMaintainOrderingWithAsyncMessages()
{
    messageFlags().add(MessageFlags::MaintainOrderingWithAsyncMessages);
}

void Encoder::wrapForTesting(UniqueRef<Encoder>&& original)
{
    ASSERT(isSyncMessage());
    ASSERT(!original->isSyncMessage());

    original->setShouldDispatchMessageWhenWaitingForSyncReply(ShouldDispatchWhenWaitingForSyncReply::Yes);

    *this << original->span();

    auto attachments = original->releaseAttachments();
    reserve(attachments.size());
    for (auto&& attachment : WTFMove(attachments))
        addAttachment(WTFMove(attachment));
}

static inline size_t roundUpToAlignment(size_t value, size_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

void Encoder::reserve(size_t size)
{
    auto oldCapacityBufferSize = capacityBuffer().size();
    if (size <= oldCapacityBufferSize)
        return;

    size_t newCapacity = roundUpToAlignment(oldCapacityBufferSize * 2, 4096);
    while (newCapacity < size)
        newCapacity *= 2;

    auto newBuffer = allocateBuffer(newCapacity);
    memcpySpan(newBuffer.mutableSpan(), span());

    m_outOfLineBuffer = WTFMove(newBuffer);
}

void Encoder::encodeHeader()
{
    *this << defaultMessageFlags;
    *this << m_messageName;
    *this << m_destinationID;
}

OptionSet<MessageFlags>& Encoder::messageFlags()
{
    // FIXME: We should probably pass an OptionSet<MessageFlags> into the Encoder constructor instead of encoding defaultMessageFlags then using this to change it later.
    static_assert(sizeof(OptionSet<MessageFlags>::StorageType) == 1, "Encoder uses the first byte of the buffer for message flags.");
    return reinterpretCastSpanStartTo<OptionSet<MessageFlags>>(capacityBuffer());
}

const OptionSet<MessageFlags>& Encoder::messageFlags() const
{
    return reinterpretCastSpanStartTo<OptionSet<MessageFlags>>(capacityBuffer());
}

std::span<uint8_t> Encoder::grow(size_t alignment, size_t size)
{
    size_t alignedSize = roundUpToAlignment(m_bufferSize, alignment);
    reserve(alignedSize + size);

    auto capacityBuffer = this->capacityBuffer();
    zeroSpan(capacityBuffer.subspan(m_bufferSize, alignedSize - m_bufferSize));

    m_bufferSize = alignedSize + size;

    return capacityBuffer.subspan(alignedSize);
}

std::span<uint8_t> Encoder::capacityBuffer()
{
    if (m_outOfLineBuffer)
        return m_outOfLineBuffer.mutableSpan();
    return m_inlineBuffer;
}

std::span<const uint8_t> Encoder::capacityBuffer() const
{
    if (m_outOfLineBuffer)
        return m_outOfLineBuffer.span();
    return m_inlineBuffer;
}

void Encoder::addAttachment(Attachment&& attachment)
{
    m_attachments.append(WTFMove(attachment));
}

Vector<Attachment> Encoder::releaseAttachments()
{
    return std::exchange(m_attachments, { });
}

bool Encoder::hasAttachments() const
{
    return !m_attachments.isEmpty();
}

} // namespace IPC
