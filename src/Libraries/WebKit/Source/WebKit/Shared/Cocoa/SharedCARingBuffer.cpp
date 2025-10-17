/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 2, 2024.
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

#if USE(MEDIATOOLBOX)
#include "SharedCARingBuffer.h"

#include "Logging.h"
#include <WebCore/CARingBuffer.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SharedCARingBufferBase);
WTF_MAKE_TZONE_ALLOCATED_IMPL(ConsumerSharedCARingBuffer);
WTF_MAKE_TZONE_ALLOCATED_IMPL(ProducerSharedCARingBuffer);

SharedCARingBufferBase::SharedCARingBufferBase(size_t bytesPerFrame, size_t frameCount, uint32_t numChannelStream, Ref<WebCore::SharedMemory> storage)
    : CARingBuffer(bytesPerFrame, frameCount, numChannelStream)
    , m_storage(WTFMove(storage))
{
}

std::unique_ptr<ConsumerSharedCARingBuffer> ConsumerSharedCARingBuffer::map(uint32_t bytesPerFrame, uint32_t numChannelStreams, ConsumerSharedCARingBuffer::Handle&& handle)
{
    auto frameCount = WTF::roundUpToPowerOfTwo(handle.frameCount);

    // Validate the parameters as they may be coming from an untrusted process.
    auto expectedStorageSize = computeSizeForBuffers(bytesPerFrame, frameCount, numChannelStreams) + sizeof(TimeBoundsBuffer);
    if (expectedStorageSize.hasOverflowed()) {
        RELEASE_LOG_FAULT(Media, "ConsumerSharedCARingBuffer::map: Overflowed when trying to compute the storage size");
        return nullptr;
    }
    auto storage = WebCore::SharedMemory::map(WTFMove(handle.memory), WebCore::SharedMemory::Protection::ReadOnly);
    if (!storage) {
        RELEASE_LOG_FAULT(Media, "ConsumerSharedCARingBuffer::map: Failed to map memory");
        return nullptr;
    }
    if (storage->size() < expectedStorageSize) {
        RELEASE_LOG_FAULT(Media, "ConsumerSharedCARingBuffer::map: Storage size is insufficient for format and frameCount");
        return nullptr;
    }

    std::unique_ptr<ConsumerSharedCARingBuffer> result { new ConsumerSharedCARingBuffer { bytesPerFrame, frameCount, numChannelStreams, storage.releaseNonNull() } };
    result->initialize();
    return result;
}

std::optional<ProducerSharedCARingBuffer::Pair> ProducerSharedCARingBuffer::allocate(const WebCore::CAAudioStreamDescription& format, size_t frameCount)
{
    frameCount = WTF::roundUpToPowerOfTwo(frameCount);
    auto bytesPerFrame = format.bytesPerFrame();
    auto numChannelStreams = format.numberOfChannelStreams();

    auto checkedSharedMemorySize = computeSizeForBuffers(bytesPerFrame, frameCount, numChannelStreams) + sizeof(TimeBoundsBuffer);
    if (checkedSharedMemorySize.hasOverflowed()) {
        RELEASE_LOG_FAULT(Media, "ProducerSharedCARingBuffer::allocate: Overflowed when trying to compute the storage size");
        return std::nullopt;
    }
    auto sharedMemory = WebCore::SharedMemory::allocate(checkedSharedMemorySize.value());
    if (!sharedMemory)
        return std::nullopt;

    auto handle = sharedMemory->createHandle(WebCore::SharedMemory::Protection::ReadOnly);
    if (!handle)
        return std::nullopt;

    new (NotNull, sharedMemory->mutableSpan().data()) TimeBoundsBuffer;
    std::unique_ptr<ProducerSharedCARingBuffer> result { new ProducerSharedCARingBuffer { bytesPerFrame, frameCount, numChannelStreams, sharedMemory.releaseNonNull() } };
    result->initialize();
    return Pair { WTFMove(result), { WTFMove(*handle), frameCount } };
}

} // namespace WebKit

#endif
