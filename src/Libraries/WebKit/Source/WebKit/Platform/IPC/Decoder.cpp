/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 1, 2024.
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
#include "Decoder.h"

#include "ArgumentCoders.h"
#include "Logging.h"
#include "MessageFlags.h"
#include <stdio.h>
#include <wtf/MallocSpan.h>
#include <wtf/ObjectIdentifier.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

namespace IPC {

static MallocSpan<uint8_t> copyBuffer(std::span<const uint8_t> buffer)
{
    auto bufferCopy = MallocSpan<uint8_t>::tryMalloc(buffer.size());
    if (!bufferCopy) {
        RELEASE_LOG_FAULT(IPC, "Decoder::copyBuffer: tryMalloc(%lu) failed", buffer.size());
        return { };
    }
    memcpySpan(bufferCopy.mutableSpan(), buffer);
    return bufferCopy;
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(Decoder);

std::unique_ptr<Decoder> Decoder::create(std::span<const uint8_t> buffer, Vector<Attachment>&& attachments)
{
    auto bufferCopy = copyBuffer(buffer);
    auto bufferCopySpan = bufferCopy.span();
    return Decoder::create(bufferCopySpan, [bufferCopy = WTFMove(bufferCopy)](auto) { }, WTFMove(attachments)); // NOLINT
}

std::unique_ptr<Decoder> Decoder::create(std::span<const uint8_t> buffer, BufferDeallocator&& bufferDeallocator, Vector<Attachment>&& attachments)
{
    ASSERT(bufferDeallocator);
    ASSERT(!!buffer.data());
    if (UNLIKELY(!buffer.data())) {
        RELEASE_LOG_FAULT(IPC, "Decoder::create() called with a null buffer (buffer size: %lu)", buffer.size_bytes());
        return nullptr;
    }
    auto decoder = std::unique_ptr<Decoder>(new Decoder(buffer, WTFMove(bufferDeallocator), WTFMove(attachments)));
    if (!decoder->isValid())
        return nullptr;
    return decoder;
}

Decoder::Decoder(std::span<const uint8_t> buffer, BufferDeallocator&& bufferDeallocator, Vector<Attachment>&& attachments)
    : m_buffer { buffer }
    , m_bufferPosition { m_buffer.begin() }
    , m_bufferDeallocator { WTFMove(bufferDeallocator) }
    , m_attachments { WTFMove(attachments) }
{
    if (UNLIKELY(reinterpret_cast<uintptr_t>(m_buffer.data()) % alignof(uint64_t))) {
        markInvalid();
        return;
    }

    auto messageFlags = decode<OptionSet<MessageFlags>>();
    if (UNLIKELY(!messageFlags))
        return;
    m_messageFlags = WTFMove(*messageFlags);

    auto messageName = decode<MessageName>();
    if (UNLIKELY(!messageName))
        return;
    m_messageName = WTFMove(*messageName);

    auto destinationID = decode<uint64_t>();
    if (UNLIKELY(!destinationID))
        return;
    // 0 is a valid destinationID but we can at least reject -1 which is the HashTable deleted value.
    if (*destinationID && !WTF::ObjectIdentifierGenericBase<uint64_t>::isValidIdentifier(*destinationID)) {
        markInvalid();
        return;
    }
    m_destinationID = WTFMove(*destinationID);
    if (messageIsSync(m_messageName)) {
        auto syncRequestID = decode<SyncRequestID>();
        if (UNLIKELY(!syncRequestID))
            return;
        m_syncRequestID = syncRequestID;
    }
}

Decoder::Decoder(std::span<const uint8_t> stream, uint64_t destinationID)
    : m_buffer { stream }
    , m_bufferPosition { m_buffer.begin() }
    , m_bufferDeallocator { nullptr }
    , m_destinationID { destinationID }
{
    // 0 is a valid destinationID but we can at least reject -1 which is the HashTable deleted value.
    if (destinationID && !WTF::ObjectIdentifierGenericBase<uint64_t>::isValidIdentifier(destinationID)) {
        markInvalid();
        return;
    }

    auto messageName = decode<MessageName>();
    if (UNLIKELY(!messageName))
        return;
    m_messageName = WTFMove(*messageName);
    if (messageIsSync(m_messageName)) {
        auto syncRequestID = decode<SyncRequestID>();
        if (UNLIKELY(!syncRequestID))
            return;
        m_syncRequestID = syncRequestID;
    }
}

Decoder::~Decoder()
{
    if (isValid())
        markInvalid();
    // FIXME: We need to dispose of the mach ports in cases of failure.
}

ShouldDispatchWhenWaitingForSyncReply Decoder::shouldDispatchMessageWhenWaitingForSyncReply() const
{
    if (m_messageFlags.contains(MessageFlags::DispatchMessageWhenWaitingForSyncReply))
        return ShouldDispatchWhenWaitingForSyncReply::Yes;
    if (m_messageFlags.contains(MessageFlags::DispatchMessageWhenWaitingForUnboundedSyncReply))
        return ShouldDispatchWhenWaitingForSyncReply::YesDuringUnboundedIPC;
    return ShouldDispatchWhenWaitingForSyncReply::No;
}

bool Decoder::shouldUseFullySynchronousModeForTesting() const
{
    return m_messageFlags.contains(MessageFlags::UseFullySynchronousModeForTesting);
}

bool Decoder::shouldMaintainOrderingWithAsyncMessages() const
{
    return m_messageFlags.contains(MessageFlags::MaintainOrderingWithAsyncMessages);
}

#if PLATFORM(MAC)
void Decoder::setImportanceAssertion(ImportanceAssertion&& assertion)
{
    m_importanceAssertion = WTFMove(assertion);
}
#endif

std::unique_ptr<Decoder> Decoder::unwrapForTesting(Decoder& decoder)
{
    ASSERT(decoder.isSyncMessage());

    auto attachments = std::exchange(decoder.m_attachments, { });

    auto wrappedMessage = decoder.decode<std::span<const uint8_t>>();
    if (!wrappedMessage)
        return nullptr;

    auto wrappedDecoder = Decoder::create(*wrappedMessage, WTFMove(attachments));
    wrappedDecoder->setIsAllowedWhenWaitingForSyncReplyOverride(true);
    return wrappedDecoder;
}

std::optional<Attachment> Decoder::takeLastAttachment()
{
    if (m_attachments.isEmpty()) {
        markInvalid();
        return std::nullopt;
    }
    return m_attachments.takeLast();
}

} // namespace IPC
