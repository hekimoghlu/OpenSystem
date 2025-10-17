/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 9, 2022.
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

#if USE(GLIB)

#include <WebCore/SharedMemory.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#if !PLATFORM(WIN)
#include <gio/gfiledescriptorbased.h>
#endif

#include <wtf/glib/GSpanExtras.h>

namespace WebKit {
namespace NetworkCache {

Data::Data(std::span<const uint8_t> data)
{
    uint8_t* copiedData = static_cast<uint8_t*>(fastMalloc(data.size()));
    memcpy(copiedData, data.data(), data.size());
    m_buffer = adoptGRef(g_bytes_new_with_free_func(copiedData, data.size(), fastFree, copiedData));
}

Data::Data(GRefPtr<GBytes>&& buffer, FileSystem::PlatformFileHandle fd)
    : m_buffer(WTFMove(buffer))
    , m_fileDescriptor(fd)
    , m_isMap(m_buffer && g_bytes_get_size(m_buffer.get()) && FileSystem::isHandleValid(fd))
{
}

Data Data::empty()
{
    return { adoptGRef(g_bytes_new(nullptr, 0)) };
}

std::span<const uint8_t> Data::span() const
{
    if (!m_buffer)
        return { };
    return WTF::span(m_buffer);
}

size_t Data::size() const
{
    return m_buffer ? g_bytes_get_size(m_buffer.get()) : 0;
}

bool Data::isNull() const
{
    return !m_buffer;
}

bool Data::apply(const Function<bool(std::span<const uint8_t>)>& applier) const
{
    if (!size())
        return false;

    return applier(span());
}

Data Data::subrange(size_t offset, size_t size) const
{
    if (!m_buffer)
        return { };

    return { adoptGRef(g_bytes_new_from_bytes(m_buffer.get(), offset, size)) };
}

Data concatenate(const Data& a, const Data& b)
{
    if (a.isNull())
        return b;
    if (b.isNull())
        return a;

    size_t size = a.size() + b.size();
    WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN // GTK/WPE port
    uint8_t* data = static_cast<uint8_t*>(fastMalloc(size));
    WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
    gsize aLength;
    const auto* aData = g_bytes_get_data(a.bytes(), &aLength);
    memcpy(data, aData, aLength);
    gsize bLength;
    const auto* bData = g_bytes_get_data(b.bytes(), &bLength);
    memcpy(data + aLength, bData, bLength);

    return { adoptGRef(g_bytes_new_with_free_func(data, size, fastFree, data)) };
}

struct MapWrapper {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;

    ~MapWrapper()
    {
        FileSystem::closeFile(fileDescriptor);
    }

    FileSystem::MappedFileData mappedFile;
    FileSystem::PlatformFileHandle fileDescriptor;
};

static void deleteMapWrapper(MapWrapper* wrapper)
{
    delete wrapper;
}

Data Data::adoptMap(FileSystem::MappedFileData&& mappedFile, FileSystem::PlatformFileHandle fd)
{
    size_t size = mappedFile.size();
    auto* map = mappedFile.span().data();
    ASSERT(map);
    ASSERT(map != MAP_FAILED);
    MapWrapper* wrapper = new MapWrapper { WTFMove(mappedFile), fd };
    return { adoptGRef(g_bytes_new_with_free_func(map, size, reinterpret_cast<GDestroyNotify>(deleteMapWrapper), wrapper)), fd };
}

RefPtr<WebCore::SharedMemory> Data::tryCreateSharedMemory() const
{
    if (isNull() || !isMap())
        return nullptr;

    int fd = FileSystem::posixFileDescriptor(m_fileDescriptor);
    gsize length;
    const auto* data = g_bytes_get_data(m_buffer.get(), &length);
    return WebCore::SharedMemory::wrapMap(const_cast<void*>(data), length, fd);
}

} // namespace NetworkCache
} // namespace WebKit

#endif // USE(GLIB)
