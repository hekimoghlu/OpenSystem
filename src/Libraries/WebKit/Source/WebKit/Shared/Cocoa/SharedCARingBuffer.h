/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 5, 2023.
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

#if USE(MEDIATOOLBOX)

#include <WebCore/CAAudioStreamDescription.h>
#include <WebCore/CARingBuffer.h>
#include <WebCore/SharedMemory.h>
#include <wtf/Atomics.h>
#include <wtf/Function.h>
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {

class SharedCARingBufferBase : public WebCore::CARingBuffer {
    WTF_MAKE_TZONE_ALLOCATED(SharedCARingBufferBase);
protected:
    SharedCARingBufferBase(size_t bytesPerFrame, size_t frameCount, uint32_t numChannelStream, Ref<WebCore::SharedMemory>);
    void* data() final { return byteCast<Byte>(m_storage->mutableSpan().subspan(sizeof(TimeBoundsBuffer)).data()); }
    TimeBoundsBuffer& timeBoundsBuffer() final { return spanReinterpretCast<TimeBoundsBuffer>(m_storage->mutableSpan().first(sizeof(TimeBoundsBuffer))).front(); }

    Ref<WebCore::SharedMemory> m_storage;
};

struct ConsumerSharedCARingBufferHandle {
    WebCore::SharedMemory::Handle memory;
    size_t frameCount { 0 };

    void takeOwnershipOfMemory(WebCore::MemoryLedger ledger) { memory.takeOwnershipOfMemory(ledger); }
};

class ConsumerSharedCARingBuffer final : public SharedCARingBufferBase {
    WTF_MAKE_TZONE_ALLOCATED(ConsumerSharedCARingBuffer);
public:
    using Handle = ConsumerSharedCARingBufferHandle;

    // FIXME: Remove this deprecated constructor.
    static std::unique_ptr<ConsumerSharedCARingBuffer> map(const WebCore::CAAudioStreamDescription& format, Handle&& handle)
    {
        return map(format.bytesPerFrame(), format.numberOfChannelStreams(), WTFMove(handle));
    }
    static std::unique_ptr<ConsumerSharedCARingBuffer> map(uint32_t bytesPerFrame, uint32_t numChannelStreams, Handle&&);
protected:
    using SharedCARingBufferBase::SharedCARingBufferBase;
};

class ProducerSharedCARingBuffer final : public SharedCARingBufferBase {
    WTF_MAKE_TZONE_ALLOCATED(ProducerSharedCARingBuffer);
public:
    struct Pair {
        std::unique_ptr<ProducerSharedCARingBuffer> producer;
        ConsumerSharedCARingBuffer::Handle consumer;
    };
    static std::optional<Pair> allocate(const WebCore::CAAudioStreamDescription& format, size_t frameCount);
protected:
    using SharedCARingBufferBase::SharedCARingBufferBase;
};

}

#endif
