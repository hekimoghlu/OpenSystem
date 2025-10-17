/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 20, 2023.
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
#include "SharedMemory.h"

#include <wtf/RefPtr.h>

namespace WebCore {

RefPtr<SharedMemory> SharedMemory::allocate(size_t size)
{
    auto handle = Win32Handle::adopt(::CreateFileMappingW(INVALID_HANDLE_VALUE, 0, PAGE_READWRITE, 0, size, 0));
    if (!handle)
        return nullptr;

    void* baseAddress = ::MapViewOfFileEx(handle.get(), FILE_MAP_ALL_ACCESS, 0, 0, size, nullptr);
    if (!baseAddress)
        return nullptr;

    RefPtr<SharedMemory> memory = adoptRef(new SharedMemory);
    memory->m_size = size;
    memory->m_data = baseAddress;
    memory->m_handle = WTFMove(handle);

    return memory;
}

static DWORD accessRights(SharedMemory::Protection protection)
{
    switch (protection) {
    case SharedMemory::Protection::ReadOnly:
        return FILE_MAP_READ;
    case SharedMemory::Protection::ReadWrite:
        return FILE_MAP_READ | FILE_MAP_WRITE;
    }

    ASSERT_NOT_REACHED();
    return 0;
}

RefPtr<SharedMemory> SharedMemory::map(Handle&& handle, Protection protection)
{
    void* data = ::MapViewOfFile(handle.m_handle.get(), accessRights(protection), 0, 0, handle.size());
    ASSERT_WITH_MESSAGE(data, "::MapViewOfFile failed with error %lu %p", ::GetLastError(), handle.m_handle.get());
    if (!data)
        return nullptr;

    RefPtr<SharedMemory> memory = adoptRef(new SharedMemory);
    memory->m_size = handle.size();
    memory->m_data = data;
    memory->m_handle = WTFMove(handle.m_handle);
    return memory;
}

SharedMemory::~SharedMemory()
{
    ASSERT(m_data);
    ASSERT(m_handle);

    ::UnmapViewOfFile(m_data);
}

auto SharedMemory::createHandle(Protection protection) -> std::optional<Handle>
{
    HANDLE processHandle = ::GetCurrentProcess();

    HANDLE duplicatedHandle;
    if (!::DuplicateHandle(processHandle, m_handle.get(), processHandle, &duplicatedHandle, accessRights(protection), FALSE, 0))
        return std::nullopt;

    return { Handle(Win32Handle::adopt(duplicatedHandle), m_size) };
}

} // namespace WebCore
