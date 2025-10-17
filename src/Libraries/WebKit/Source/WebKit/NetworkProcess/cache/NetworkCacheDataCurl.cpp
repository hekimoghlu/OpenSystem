/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 11, 2022.
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
#include "NetworkCacheData.h"

#include <WebCore/SharedMemory.h>
#include <wtf/StdLibExtras.h>

namespace WebKit {
namespace NetworkCache {

Data::Data(std::span<const uint8_t> data)
    : m_buffer(Box<std::variant<Vector<uint8_t>, FileSystem::MappedFileData>>::create(Vector<uint8_t>(data.size())))
{
    memcpy(std::get<Vector<uint8_t>>(*m_buffer).data(), data.data(), data.size());
}

Data::Data(std::variant<Vector<uint8_t>, FileSystem::MappedFileData>&& data)
    : m_buffer(Box<std::variant<Vector<uint8_t>, FileSystem::MappedFileData>>::create(WTFMove(data)))
    , m_isMap(std::holds_alternative<FileSystem::MappedFileData>(*m_buffer))
{
}

Data Data::empty()
{
    Vector<uint8_t> buffer;
    return { WTFMove(buffer) };
}

std::span<const uint8_t> Data::span() const
{
    if (!m_buffer)
        return { };

    return WTF::switchOn(*m_buffer,
        [](const Vector<uint8_t>& buffer) { return buffer.span(); },
        [](const FileSystem::MappedFileData& mappedFile) { return mappedFile.span(); }
    );
}

size_t Data::size() const
{
    if (!m_buffer)
        return 0;
    return WTF::switchOn(*m_buffer,
        [](const Vector<uint8_t>& buffer) -> size_t { return buffer.size(); },
        [](const FileSystem::MappedFileData& mappedFile) -> size_t { return mappedFile.size(); }
    );
}

bool Data::isNull() const
{
    return !m_buffer;
}

bool Data::apply(const Function<bool(std::span<const uint8_t>)>& applier) const
{
    if (isEmpty())
        return false;

    return applier(span());
}

Data Data::subrange(size_t offset, size_t size) const
{
    if (!m_buffer)
        return { };

    return span().subspan(offset, size);
}

Data concatenate(const Data& a, const Data& b)
{
    if (a.isNull())
        return b;
    if (b.isNull())
        return a;

    Vector<uint8_t> buffer(a.size() + b.size());
    memcpySpan(buffer.mutableSpan(), a.span());
    memcpySpan(buffer.mutableSpan().subspan(a.size()), b.span());
    return Data(WTFMove(buffer));
}

Data Data::adoptMap(FileSystem::MappedFileData&& mappedFile, FileSystem::PlatformFileHandle fd)
{
    ASSERT(mappedFile);
    FileSystem::closeFile(fd);

    return { WTFMove(mappedFile) };
}

#if ENABLE(SHAREABLE_RESOURCE) && OS(WINDOWS)
RefPtr<WebCore::SharedMemory> Data::tryCreateSharedMemory() const
{
    if (isNull() || !isMap())
        return nullptr;

    auto newHandle = Win32Handle { std::get<FileSystem::MappedFileData>(*m_buffer).fileMapping() };
    if (!newHandle)
        return nullptr;

    return WebCore::SharedMemory::map({ WTFMove(newHandle), size() }, WebCore::SharedMemory::Protection::ReadOnly);
}
#endif

} // namespace NetworkCache
} // namespace WebKit
