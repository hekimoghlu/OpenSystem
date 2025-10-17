/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 10, 2024.
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
#include "RemoteBufferProxy.h"

#if ENABLE(GPU_PROCESS)

#include "RemoteBufferMessages.h"
#include "WebGPUConvertToBackingContext.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit::WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(RemoteBufferProxy);

RemoteBufferProxy::RemoteBufferProxy(RemoteDeviceProxy& parent, ConvertToBackingContext& convertToBackingContext, WebGPUIdentifier identifier, bool mappedAtCreation)
    : m_backing(identifier)
    , m_convertToBackingContext(convertToBackingContext)
    , m_parent(parent)
    , m_mapModeFlags(mappedAtCreation ? WebCore::WebGPU::MapModeFlags(WebCore::WebGPU::MapMode::Write) : WebCore::WebGPU::MapModeFlags())
{
}

RemoteBufferProxy::~RemoteBufferProxy()
{
    auto sendResult = send(Messages::RemoteBuffer::Destruct());
    UNUSED_VARIABLE(sendResult);
}

void RemoteBufferProxy::mapAsync(WebCore::WebGPU::MapModeFlags mapModeFlags, WebCore::WebGPU::Size64 offset, std::optional<WebCore::WebGPU::Size64> size, CompletionHandler<void(bool)>&& callback)
{
    auto sendResult = sendWithAsyncReply(Messages::RemoteBuffer::MapAsync(mapModeFlags, offset, size), [callback = WTFMove(callback), mapModeFlags, protectedThis = Ref { *this }](auto success) mutable {
        if (!success)
            return callback(false);
        protectedThis->m_mapModeFlags = mapModeFlags;
        callback(true);
    });
    UNUSED_PARAM(sendResult);
}

static bool offsetOrSizeExceedsBounds(size_t dataSize, WebCore::WebGPU::Size64 offset, std::optional<WebCore::WebGPU::Size64> requestedSize)
{
    return offset >= dataSize || (requestedSize.has_value() && requestedSize.value() + offset > dataSize);
}

void RemoteBufferProxy::getMappedRange(WebCore::WebGPU::Size64 offset, std::optional<WebCore::WebGPU::Size64> size, Function<void(std::span<uint8_t>)>&& callback)
{
    // FIXME: Implement error handling.
    auto sendResult = sendSync(Messages::RemoteBuffer::GetMappedRange(offset, size));
    auto [data] = sendResult.takeReplyOr(std::nullopt);

    if (!data || !data->data() || offsetOrSizeExceedsBounds(data->size(), offset, size)) {
        callback({ });
        return;
    }

    callback(data->mutableSpan().subspan(offset));
}

std::span<uint8_t> RemoteBufferProxy::getBufferContents()
{
    RELEASE_ASSERT_NOT_REACHED();
}

void RemoteBufferProxy::copyFrom(std::span<const uint8_t> span, size_t offset)
{
    if (!m_mapModeFlags.contains(WebCore::WebGPU::MapMode::Write))
        return;

    auto sharedMemory = WebCore::SharedMemory::copySpan(span);
    std::optional<WebCore::SharedMemoryHandle> handle;
    if (sharedMemory)
        handle = sharedMemory->createHandle(WebCore::SharedMemory::Protection::ReadOnly);
    auto sendResult = sendWithAsyncReply(Messages::RemoteBuffer::Copy(WTFMove(handle), offset), [sharedMemory = sharedMemory.copyRef(), handleHasValue = handle.has_value()](auto) mutable {
        RELEASE_ASSERT(sharedMemory.get() || !handleHasValue);
    });
    UNUSED_VARIABLE(sendResult);
}

void RemoteBufferProxy::unmap()
{
    auto sendResult = send(Messages::RemoteBuffer::Unmap());
    UNUSED_VARIABLE(sendResult);

    m_mapModeFlags = { };
}

void RemoteBufferProxy::destroy()
{
    auto sendResult = send(Messages::RemoteBuffer::Destroy());
    UNUSED_VARIABLE(sendResult);
}

void RemoteBufferProxy::setLabelInternal(const String& label)
{
    auto sendResult = send(Messages::RemoteBuffer::SetLabel(label));
    UNUSED_VARIABLE(sendResult);
}

} // namespace WebKit::WebGPU

#endif // ENABLE(GPU_PROCESS)
