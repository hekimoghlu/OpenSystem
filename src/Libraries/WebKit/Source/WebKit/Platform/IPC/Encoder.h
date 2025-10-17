/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 11, 2024.
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

#include "Attachment.h"
#include "MessageNames.h"
#include <WebCore/PlatformExportMacros.h>
#include <WebCore/SharedBuffer.h>
#include <wtf/Forward.h>
#include <wtf/MallocSpan.h>
#include <wtf/OptionSet.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

#if OS(DARWIN)
namespace WTF {
struct Mmap;
}
#endif

namespace IPC {

enum class MessageFlags : uint8_t;
enum class ShouldDispatchWhenWaitingForSyncReply : uint8_t;

template<typename, typename> struct ArgumentCoder;

class Encoder final {
    WTF_MAKE_TZONE_ALLOCATED(Encoder);
public:
    Encoder(MessageName, uint64_t destinationID);
    ~Encoder();

    Encoder(const Encoder&) = delete;
    Encoder(Encoder&&) = delete;
    Encoder& operator=(const Encoder&) = delete;
    Encoder& operator=(Encoder&&) = delete;

    ReceiverName messageReceiverName() const { return receiverName(m_messageName); }
    MessageName messageName() const { return m_messageName; }
    uint64_t destinationID() const { return m_destinationID; }

    bool isSyncMessage() const { return messageIsSync(messageName()); }

    void setShouldDispatchMessageWhenWaitingForSyncReply(ShouldDispatchWhenWaitingForSyncReply);

    bool isFullySynchronousModeForTesting() const;
    void setFullySynchronousModeForTesting();
    void setShouldMaintainOrderingWithAsyncMessages();
    bool isAllowedWhenWaitingForSyncReply() const { return messageAllowedWhenWaitingForSyncReply(messageName()) || isFullySynchronousModeForTesting(); }
    bool isAllowedWhenWaitingForUnboundedSyncReply() const { return messageAllowedWhenWaitingForUnboundedSyncReply(messageName()); }

    void wrapForTesting(UniqueRef<Encoder>&&);

    template<typename T, size_t Extent> void encodeSpan(std::span<T, Extent>);
    template<typename T> void encodeObject(const T&);

    template<typename T>
    Encoder& operator<<(T&& t)
    {
        ArgumentCoder<std::remove_cvref_t<T>, void>::encode(*this, std::forward<T>(t));
        return *this;
    }

    Encoder& operator<<(Attachment&& attachment)
    {
        addAttachment(WTFMove(attachment));
        return *this;
    }

    std::span<uint8_t> mutableSpan() { return capacityBuffer().first(m_bufferSize); }
    std::span<const uint8_t> span() const { return capacityBuffer().first(m_bufferSize); }

    void addAttachment(Attachment&&);
    Vector<Attachment> releaseAttachments();
    void reserve(size_t);

    static constexpr bool isIPCEncoder = true;

private:
    std::span<uint8_t> grow(size_t alignment, size_t);

    std::span<uint8_t> capacityBuffer();
    std::span<const uint8_t> capacityBuffer() const;

    bool hasAttachments() const;

    void encodeHeader();
    const OptionSet<MessageFlags>& messageFlags() const;
    OptionSet<MessageFlags>& messageFlags();

    void freeBufferIfNecessary();

    MessageName m_messageName;
    uint64_t m_destinationID;

#if OS(DARWIN)
    MallocSpan<uint8_t, WTF::Mmap> m_outOfLineBuffer;
#else
    MallocSpan<uint8_t> m_outOfLineBuffer;
#endif
    std::array<uint8_t, 512> m_inlineBuffer;

    size_t m_bufferSize { 0 };

    Vector<Attachment> m_attachments;
};

template<typename T, size_t Extent>
inline void Encoder::encodeSpan(std::span<T, Extent> span)
{
    auto bytes = asBytes(span);
    constexpr size_t alignment = alignof(T);
    ASSERT(!(reinterpret_cast<uintptr_t>(bytes.data()) % alignment));

    auto buffer = grow(alignment, bytes.size());
    memcpySpan(buffer, bytes);
}

template<typename T>
inline void Encoder::encodeObject(const T& object)
{
    static_assert(std::is_trivially_copyable_v<T>);
    encodeSpan(unsafeMakeSpan(std::addressof(object), 1));
}

} // namespace IPC
