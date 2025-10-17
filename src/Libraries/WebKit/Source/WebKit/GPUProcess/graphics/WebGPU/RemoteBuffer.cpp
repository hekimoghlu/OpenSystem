/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 28, 2022.
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
#include "RemoteBuffer.h"

#if ENABLE(GPU_PROCESS)

#include "RemoteBufferMessages.h"
#include "StreamServerConnection.h"
#include "WebGPUObjectHeap.h"

#include <WebCore/SharedMemory.h>
#include <wtf/CheckedArithmetic.h>
#include <wtf/TZoneMalloc.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteBuffer);

RemoteBuffer::RemoteBuffer(WebCore::WebGPU::Buffer& buffer, WebGPU::ObjectHeap& objectHeap, Ref<IPC::StreamServerConnection>&& streamConnection, RemoteGPU& gpu, bool mappedAtCreation, WebGPUIdentifier identifier)
    : m_backing(buffer)
    , m_objectHeap(objectHeap)
    , m_streamConnection(WTFMove(streamConnection))
    , m_gpu(gpu)
    , m_identifier(identifier)
    , m_isMapped(mappedAtCreation)
    , m_mapModeFlags(mappedAtCreation ? WebCore::WebGPU::MapModeFlags(WebCore::WebGPU::MapMode::Write) : WebCore::WebGPU::MapModeFlags())
{
    protectedStreamConnection()->startReceivingMessages(*this, Messages::RemoteBuffer::messageReceiverName(), m_identifier.toUInt64());
}

RemoteBuffer::~RemoteBuffer() = default;

void RemoteBuffer::stopListeningForIPC()
{
    protectedStreamConnection()->stopReceivingMessages(Messages::RemoteBuffer::messageReceiverName(), m_identifier.toUInt64());
}

void RemoteBuffer::mapAsync(WebCore::WebGPU::MapModeFlags mapModeFlags, WebCore::WebGPU::Size64 offset, std::optional<WebCore::WebGPU::Size64> size, CompletionHandler<void(bool)>&& callback)
{
    m_isMapped = true;
    m_mapModeFlags = mapModeFlags;

    protectedBacking()->mapAsync(mapModeFlags, offset, size, [protectedThis = Ref<RemoteBuffer>(*this), callback = WTFMove(callback)] (bool success) mutable {
        if (!success) {
            callback(false);
            return;
        }

        callback(true);
    });
}

void RemoteBuffer::getMappedRange(WebCore::WebGPU::Size64 offset, std::optional<WebCore::WebGPU::Size64> size, CompletionHandler<void(std::optional<Vector<uint8_t>>&&)>&& callback)
{
    protectedBacking()->getMappedRange(offset, size, [&] (auto mappedRange) {
        m_isMapped = true;
        callback(mappedRange);
    });
}

void RemoteBuffer::unmap()
{
    if (m_isMapped)
        protectedBacking()->unmap();
    m_isMapped = false;
    m_mapModeFlags = { };
}

void RemoteBuffer::copy(std::optional<WebCore::SharedMemoryHandle>&& dataHandle, size_t offset, CompletionHandler<void(bool)>&& completionHandler)
{
    auto sharedData = dataHandle ? WebCore::SharedMemory::map(WTFMove(*dataHandle), WebCore::SharedMemory::Protection::ReadOnly) : nullptr;
    auto data = sharedData ? sharedData->span() : std::span<const uint8_t> { };
    if (!m_isMapped || !m_mapModeFlags.contains(WebCore::WebGPU::MapMode::Write)) {
        completionHandler(false);
        return;
    }

#if !ENABLE(WEBGPU_SWIFT)
    auto buffer = protectedBacking()->getBufferContents();
    if (buffer.empty()) {
        completionHandler(false);
        return;
    }

    auto endOffset = checkedSum<size_t>(offset, data.size());
    if (endOffset.hasOverflowed() || endOffset.value() > buffer.size()) {
        completionHandler(false);
        return;
    }

    memcpySpan(buffer.subspan(offset), data);
#else
    backing().copyFrom(data, offset);
#endif
    completionHandler(true);

}

void RemoteBuffer::destroy()
{
    unmap();
    protectedBacking()->destroy();
}

void RemoteBuffer::destruct()
{
    Ref { m_objectHeap.get() }->removeObject(m_identifier);
}

void RemoteBuffer::setLabel(String&& label)
{
    protectedBacking()->setLabel(WTFMove(label));
}

Ref<WebCore::WebGPU::Buffer> RemoteBuffer::protectedBacking()
{
    return m_backing;
}

Ref<IPC::StreamServerConnection> RemoteBuffer::protectedStreamConnection() const
{
    return m_streamConnection;
}

} // namespace WebKit

#endif // ENABLE(GPU_PROCESS)
