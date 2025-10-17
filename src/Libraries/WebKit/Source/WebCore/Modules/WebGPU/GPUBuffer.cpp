/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 25, 2023.
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
#include "GPUBuffer.h"

#include "GPUDevice.h"
#include "JSDOMPromiseDeferred.h"
#include "JSGPUBufferMapState.h"

namespace WebCore {

GPUBuffer::~GPUBuffer() = default;

GPUBuffer::GPUBuffer(Ref<WebGPU::Buffer>&& backing, size_t bufferSize, GPUBufferUsageFlags usage, bool mappedAtCreation, GPUDevice& device)
    : m_backing(WTFMove(backing))
    , m_bufferSize(bufferSize)
    , m_usage(usage)
    , m_mapState(mappedAtCreation ? GPUBufferMapState::Mapped : GPUBufferMapState::Unmapped)
    , m_device(device)
    , m_mappedAtCreation(mappedAtCreation)
{
    if (mappedAtCreation)
        m_mappedRangeSize = m_bufferSize;
}

String GPUBuffer::label() const
{
    return m_backing->label();
}

void GPUBuffer::setLabel(String&& label)
{
    m_backing->setLabel(WTFMove(label));
}

void GPUBuffer::mapAsync(GPUMapModeFlags mode, std::optional<GPUSize64> offset, std::optional<GPUSize64> size, MapAsyncPromise&& promise)
{
    if (m_pendingMapPromise) {
        promise.reject(Exception { ExceptionCode::OperationError, "pendingMapPromise"_s });
        return;
    }

    if (m_mapState == GPUBufferMapState::Unmapped)
        m_mapState = GPUBufferMapState::Pending;

    m_pendingMapPromise = promise;
    // FIXME: Should this capture a weak pointer to |this| instead?
    m_backing->mapAsync(convertMapModeFlagsToBacking(mode), offset.value_or(0), size, [promise = WTFMove(promise), protectedThis = Ref { *this }, offset, size](bool success) mutable {
        if (!protectedThis->m_pendingMapPromise) {
            if (protectedThis->m_destroyed)
                promise.reject(Exception { ExceptionCode::OperationError, "buffer destroyed during mapAsync"_s });
            else
                promise.resolve(nullptr);
            return;
        }

        protectedThis->m_pendingMapPromise = std::nullopt;
        if (success) {
            protectedThis->m_mapState = GPUBufferMapState::Mapped;
            protectedThis->m_mappedRangeOffset = offset.value_or(0);
            protectedThis->m_mappedRangeSize = size.value_or(protectedThis->m_bufferSize - protectedThis->m_mappedRangeOffset);
            promise.resolve(nullptr);
        } else {
            if (protectedThis->m_mapState == GPUBufferMapState::Pending)
                protectedThis->m_mapState = GPUBufferMapState::Unmapped;

            promise.reject(Exception { ExceptionCode::OperationError, "map async was not successful"_s });
        }
    });
}

static auto makeArrayBuffer(std::variant<std::span<const uint8_t>, size_t> source, size_t offset, auto& cachedArrayBuffers, auto& device, auto& buffer)
{
    RefPtr<ArrayBuffer> arrayBuffer;
    std::visit(WTF::makeVisitor([&](std::span<const uint8_t> source) {
        arrayBuffer = ArrayBuffer::create(source);
    }, [&](size_t numberOfElements) {
        arrayBuffer = ArrayBuffer::create(numberOfElements, 1);
    }), source);

    cachedArrayBuffers.append({ arrayBuffer.get(), offset });
    cachedArrayBuffers.last().buffer->pin();
    if (device)
        device->addBufferToUnmap(buffer);
    return arrayBuffer;
}

static bool containsRange(size_t offset, size_t endOffset, const auto& mappedRanges, const auto& mappedPoints)
{
    if (offset == endOffset) {
        if (mappedPoints.contains(offset))
            return true;

        for (auto& range : mappedRanges) {
            if (range.begin() < offset && offset < range.end())
                return true;
        }
        return false;
    }

    if (mappedRanges.overlaps({ offset, endOffset }))
        return true;

    for (auto& i : mappedPoints) {
        if (offset < i && i < endOffset)
            return true;
    }

    return false;
}

ExceptionOr<Ref<JSC::ArrayBuffer>> GPUBuffer::getMappedRange(std::optional<GPUSize64> optionalOffset, std::optional<GPUSize64> optionalSize)
{
    if (m_mapState != GPUBufferMapState::Mapped || m_destroyed)
        return Exception { ExceptionCode::OperationError, "not mapped or destroyed"_s };

    auto offset = optionalOffset.value_or(0);
    if (offset > m_bufferSize)
        return Exception { ExceptionCode::OperationError, "offset > bufferSize"_s };

    auto size = optionalSize.value_or(m_bufferSize - offset);
    auto checkedEndOffset = checkedSum<uint64_t>(offset, size);
    if (checkedEndOffset.hasOverflowed())
        return Exception { ExceptionCode::OperationError, "has overflowed"_s };

    auto endOffset = checkedEndOffset.value();
    if (offset % 8)
        return Exception { ExceptionCode::OperationError, "validation failed offset % 8"_s };

    if (size % 4)
        return Exception { ExceptionCode::OperationError, "validation failed size % 4"_s };

    if (offset < m_mappedRangeOffset)
        return Exception { ExceptionCode::OperationError, "validation failed offset < m_mappedRangeOffset"_s };

    if (endOffset > m_mappedRangeSize + m_mappedRangeOffset)
        return Exception { ExceptionCode::OperationError, "getMappedRangeFailed because offset + size > mappedRangeSize + mappedRangeOffset"_s };

    if (endOffset > m_bufferSize)
        return Exception { ExceptionCode::OperationError, "validation failed endOffset > bufferSie"_s };

    if (containsRange(offset, endOffset, m_mappedRanges, m_mappedPoints))
        return Exception { ExceptionCode::OperationError, "validation failed - containsRange"_s };

    if (offset == endOffset)
        m_mappedPoints.add(offset);
    else {
        m_mappedRanges.add({ static_cast<size_t>(offset), static_cast<size_t>(endOffset) });
        m_mappedRanges.compact();
    }

    RefPtr<JSC::ArrayBuffer> result;
    m_backing->getMappedRange(offset, size, [&] (auto mappedRange) {
        if (!mappedRange.data()) {
            m_arrayBuffers.clear();
            if (m_mappedAtCreation || !size)
                result = makeArrayBuffer(0U /* numberOfElements */, 0 /* offset */, m_arrayBuffers, m_device, *this);

            return;
        }

        result = makeArrayBuffer(mappedRange.first(size), offset, m_arrayBuffers, m_device, *this);
    });

    if (!result)
        return Exception { ExceptionCode::OperationError, "getMappedRange failed"_s };

    return result.releaseNonNull();
}

void GPUBuffer::unmap(ScriptExecutionContext& scriptExecutionContext)
{
    internalUnmap(scriptExecutionContext);
    if (m_device)
        m_device->removeBufferToUnmap(*this);
}

void GPUBuffer::internalUnmap(ScriptExecutionContext& scriptExecutionContext)
{
    m_mappedAtCreation = false;
    m_mappedRangeOffset = 0;
    m_mappedRangeSize = 0;
    m_mappedRanges.clear();
    m_mappedPoints.clear();
    if (m_pendingMapPromise) {
        m_pendingMapPromise->reject(Exception { ExceptionCode::AbortError });
        m_pendingMapPromise = std::nullopt;
    }

    m_mapState = GPUBufferMapState::Unmapped;

    for (auto& arrayBufferAndOffset : m_arrayBuffers) {
        auto& arrayBuffer = arrayBufferAndOffset.buffer;
        if (arrayBuffer && arrayBuffer->data() && arrayBuffer->byteLength()) {
            m_backing->copyFrom(arrayBuffer->span(), arrayBufferAndOffset.offset);
            JSC::ArrayBufferContents emptyBuffer;
            arrayBuffer->unpin();
            arrayBuffer->transferTo(scriptExecutionContext.vm(), emptyBuffer);
        }
    }

    m_backing->unmap();
    m_arrayBuffers.clear();
}

void GPUBuffer::destroy(ScriptExecutionContext& scriptExecutionContext)
{
    m_destroyed = true;
    internalUnmap(scriptExecutionContext);
    m_bufferSize = 0;
    m_backing->destroy();
}

}
