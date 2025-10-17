/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 13, 2024.
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
#include "ScriptBuffer.h"

#include <wtf/StdLibExtras.h>
#include <wtf/text/StringBuilder.h>

namespace WebCore {

#if ENABLE(SHAREABLE_RESOURCE) && PLATFORM(COCOA)
static std::optional<ShareableResource::Handle> tryConvertToShareableResourceHandle(const ScriptBuffer& script)
{
    if (!script.containsSingleFileMappedSegment())
        return std::nullopt;

    auto& segment = script.buffer()->begin()->segment;
    auto sharedMemory = SharedMemory::wrapMap(segment->span(), SharedMemory::Protection::ReadOnly);
    if (!sharedMemory)
        return std::nullopt;

    auto shareableResource = ShareableResource::create(sharedMemory.releaseNonNull(), 0, segment->size());
    if (!shareableResource)
        return std::nullopt;

    return shareableResource->createHandle();
}
#endif

ScriptBuffer::ScriptBuffer(const String& string)
{
    append(string);
}

ScriptBuffer ScriptBuffer::empty()
{
    return ScriptBuffer { SharedBuffer::create() };
}

String ScriptBuffer::toString() const
{
    if (!m_buffer)
        return String();

    StringBuilder builder;
    m_buffer.get()->forEachSegment([&](auto segment) {
        builder.append(byteCast<char8_t>(segment));
    });
    return builder.toString();
}

bool ScriptBuffer::containsSingleFileMappedSegment() const
{
    return m_buffer && m_buffer.get()->hasOneSegment() && m_buffer.get()->begin()->segment->containsMappedFileData();
}

void ScriptBuffer::append(const String& string)
{
    if (string.isEmpty())
        return;
    auto result = string.tryGetUTF8([&](std::span<const char8_t> span) -> bool {
        m_buffer.append(span);
        return true;
    });
    RELEASE_ASSERT(result);
}

void ScriptBuffer::append(const FragmentedSharedBuffer& buffer)
{
    m_buffer.append(buffer);
}

std::optional<ScriptBuffer> ScriptBuffer::fromIPCData(IPCData&& ipcData)
{
#if ENABLE(SHAREABLE_RESOURCE) && PLATFORM(COCOA)
    return WTF::switchOn(WTFMove(ipcData), [](ShareableResourceHandle&& handle) -> std::optional<ScriptBuffer> {
        if (RefPtr buffer = WTFMove(handle).tryWrapInSharedBuffer())
            return ScriptBuffer { WTFMove(buffer) };
        return std::nullopt;
    }, [](RefPtr<FragmentedSharedBuffer>&& buffer) -> std::optional<ScriptBuffer> {
        return ScriptBuffer { WTFMove(buffer) };
    });
#else
    return ScriptBuffer { WTFMove(ipcData) };
#endif
}

auto ScriptBuffer::ipcData() const -> IPCData
{
#if ENABLE(SHAREABLE_RESOURCE) && PLATFORM(COCOA)
    if (auto handle = tryConvertToShareableResourceHandle(*this))
        return { WTFMove(*handle) };
#endif
    return m_buffer.get();
}

bool operator==(const ScriptBuffer& a, const ScriptBuffer& b)
{
    if (a.buffer() == b.buffer())
        return true;
    if (!a.buffer() || !b.buffer())
        return false;
    return *a.buffer() == *b.buffer();
}

} // namespace WebCore
