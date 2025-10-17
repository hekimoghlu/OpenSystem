/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 29, 2023.
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
#import "config.h"
#import "NetworkCacheData.h"

#import <WebCore/SharedMemory.h>
#import <dispatch/dispatch.h>
#import <sys/mman.h>
#import <sys/stat.h>
#import <wtf/cocoa/SpanCocoa.h>

namespace WebKit {
namespace NetworkCache {

Data::Data(std::span<const uint8_t> data)
    : m_dispatchData(adoptOSObject(dispatch_data_create(data.data(), data.size(), nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT)))
{
}

Data::Data(OSObjectPtr<dispatch_data_t>&& dispatchData, Backing backing)
    : m_dispatchData(WTFMove(dispatchData))
    , m_isMap(backing == Backing::Map && dispatch_data_get_size(m_dispatchData.get()))
{
}

Data Data::empty()
{
    return { OSObjectPtr<dispatch_data_t> { dispatch_data_empty } };
}

std::span<const uint8_t> Data::span() const
{
    if (!m_data.data() && m_dispatchData) {
        const void* data = nullptr;
        size_t size = 0;
        m_dispatchData = adoptOSObject(dispatch_data_create_map(m_dispatchData.get(), &data, &size));
        m_data = unsafeMakeSpan(static_cast<const uint8_t*>(data), size);
    }
    return m_data;
}

size_t Data::size() const
{
    if (!m_data.data() && m_dispatchData)
        return dispatch_data_get_size(m_dispatchData.get());
    return m_data.size();
}

bool Data::isNull() const
{
    return !m_dispatchData;
}

bool Data::apply(const Function<bool(std::span<const uint8_t>)>& applier) const
{
    if (!size())
        return false;
    return dispatch_data_apply_span(m_dispatchData.get(), applier);
}

Data Data::subrange(size_t offset, size_t size) const
{
    return { adoptOSObject(dispatch_data_create_subrange(dispatchData(), offset, size)) };
}

Data concatenate(const Data& a, const Data& b)
{
    if (a.isNull())
        return b;
    if (b.isNull())
        return a;
    return { adoptOSObject(dispatch_data_create_concat(a.dispatchData(), b.dispatchData())) };
}

Data Data::adoptMap(FileSystem::MappedFileData&& mappedFile, FileSystem::PlatformFileHandle fd)
{
    auto span = mappedFile.leakHandle();
    ASSERT(span.data());
    ASSERT(span.data() != MAP_FAILED);
    FileSystem::closeFile(fd);
    auto bodyMap = adoptOSObject(dispatch_data_create(span.data(), span.size(), dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), [span] {
        munmap(span.data(), span.size());
    }));
    return { WTFMove(bodyMap), Data::Backing::Map };
}

RefPtr<WebCore::SharedMemory> Data::tryCreateSharedMemory() const
{
    if (isNull() || !isMap())
        return nullptr;

    return WebCore::SharedMemory::wrapMap(span(), WebCore::SharedMemory::Protection::ReadOnly);
}

}
}
