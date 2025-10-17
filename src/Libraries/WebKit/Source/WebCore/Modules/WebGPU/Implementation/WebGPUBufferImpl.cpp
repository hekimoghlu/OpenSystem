/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 31, 2023.
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
#include "WebGPUBufferImpl.h"

#if HAVE(WEBGPU_IMPLEMENTATION)

#include "WebGPUConvertToBackingContext.h"
#include <WebGPU/WebGPUExt.h>
#include <wtf/BlockPtr.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore::WebGPU {

WTF_MAKE_TZONE_ALLOCATED_IMPL(BufferImpl);

BufferImpl::BufferImpl(WebGPUPtr<WGPUBuffer>&& buffer, ConvertToBackingContext& convertToBackingContext)
    : m_backing(WTFMove(buffer))
    , m_convertToBackingContext(convertToBackingContext)
{
}

BufferImpl::~BufferImpl() = default;

static Size64 getMappedSize(WGPUBuffer buffer, std::optional<Size64> size, Size64 offset)
{
    if (size.has_value())
        return size.value();

    auto bufferSize = wgpuBufferGetInitialSize(buffer);
    return bufferSize > offset ? (bufferSize - offset) : 0;
}

static void mapAsyncCallback(WGPUBufferMapAsyncStatus status, void* userdata)
{
    auto block = reinterpret_cast<void(^)(WGPUBufferMapAsyncStatus)>(userdata);
    block(status);
    Block_release(block); // Block_release is matched with Block_copy below in BufferImpl::mapAsync().
}

void BufferImpl::mapAsync(MapModeFlags mapModeFlags, Size64 offset, std::optional<Size64> size, CompletionHandler<void(bool)>&& callback)
{
    auto backingMapModeFlags = m_convertToBackingContext->convertMapModeFlagsToBacking(mapModeFlags);
    auto usedSize = getMappedSize(m_backing.get(), size, offset);

    // FIXME: Check the casts.
    auto blockPtr = makeBlockPtr([callback = WTFMove(callback)](WGPUBufferMapAsyncStatus status) mutable {
        callback(status == WGPUBufferMapAsyncStatus_Success);
    });
    wgpuBufferMapAsync(m_backing.get(), backingMapModeFlags, static_cast<size_t>(offset), static_cast<size_t>(usedSize), &mapAsyncCallback, Block_copy(blockPtr.get())); // Block_copy is matched with Block_release above in mapAsyncCallback().
}

void BufferImpl::getMappedRange(Size64 offset, std::optional<Size64> size, Function<void(std::span<uint8_t>)>&& callback)
{
    auto usedSize = getMappedSize(m_backing.get(), size, offset);

    auto pointer = wgpuBufferGetMappedRange(m_backing.get(), static_cast<size_t>(offset), static_cast<size_t>(usedSize)).data();
    auto bufferSize = wgpuBufferGetInitialSize(m_backing.get());
    size_t actualSize = pointer ? static_cast<size_t>(bufferSize) : 0;
    size_t actualOffset = pointer ? static_cast<size_t>(offset) : 0;
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
    callback(unsafeMakeSpan(static_cast<uint8_t*>(pointer) - actualOffset, actualSize));
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
}

std::span<uint8_t> BufferImpl::getBufferContents()
{
    if (!m_backing.get())
        return { };

    return wgpuBufferGetBufferContents(m_backing.get());
}

#if ENABLE(WEBGPU_SWIFT)
void BufferImpl::copyFrom(std::span<const uint8_t> data, size_t offset)
{
    RELEASE_ASSERT(backing());
    return wgpuBufferCopy(backing(), data, offset);
}
#else
void BufferImpl::copyFrom(std::span<const uint8_t>, size_t)
{
    RELEASE_ASSERT_NOT_REACHED();
}
#endif

void BufferImpl::unmap()
{
    wgpuBufferUnmap(m_backing.get());
}

void BufferImpl::destroy()
{
    wgpuBufferDestroy(m_backing.get());
}

void BufferImpl::setLabelInternal(const String& label)
{
    wgpuBufferSetLabel(m_backing.get(), label.utf8().data());
}

} // namespace WebCore::WebGPU

#endif // HAVE(WEBGPU_IMPLEMENTATION)
