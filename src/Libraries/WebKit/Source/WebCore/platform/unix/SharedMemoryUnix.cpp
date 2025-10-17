/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 28, 2024.
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

#if USE(UNIX_DOMAIN_SOCKETS)

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <wtf/Assertions.h>
#include <wtf/CryptographicallyRandomNumber.h>
#include <wtf/SafeStrerror.h>
#include <wtf/UniStdExtras.h>
#include <wtf/text/CString.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/WTFString.h>

#if HAVE(LINUX_MEMFD_H)
#include <linux/memfd.h>
#include <sys/syscall.h>
#endif

namespace WebCore {

SharedMemoryHandle::SharedMemoryHandle(const SharedMemoryHandle& handle)
{
    m_handle = handle.m_handle.duplicate();
    m_size = handle.m_size;
}

UnixFileDescriptor SharedMemoryHandle::releaseHandle()
{
    return WTFMove(m_handle);
}

static inline int accessModeMMap(SharedMemory::Protection protection)
{
    switch (protection) {
    case SharedMemory::Protection::ReadOnly:
        return PROT_READ;
    case SharedMemory::Protection::ReadWrite:
        return PROT_READ | PROT_WRITE;
    }

    ASSERT_NOT_REACHED();
    return PROT_READ | PROT_WRITE;
}

static UnixFileDescriptor createSharedMemory()
{
    int fileDescriptor = -1;

#if HAVE(LINUX_MEMFD_H)
    static bool isMemFdAvailable = true;
    if (isMemFdAvailable) {
        do {
            fileDescriptor = syscall(__NR_memfd_create, "WebKitSharedMemory", MFD_CLOEXEC);
        } while (fileDescriptor == -1 && errno == EINTR);

        if (fileDescriptor != -1)
            return UnixFileDescriptor { fileDescriptor, UnixFileDescriptor::Adopt };

        if (errno != ENOSYS)
            return { };

        isMemFdAvailable = false;
    }
#endif

#if HAVE(SHM_ANON)
    do {
        fileDescriptor = shm_open(SHM_ANON, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
    } while (fileDescriptor == -1 && errno == EINTR);
#else
    CString tempName;
    for (int tries = 0; fileDescriptor == -1 && tries < 10; ++tries) {
        auto name = makeString("/WK2SharedMemory."_s, cryptographicallyRandomNumber<unsigned>());
        tempName = name.utf8();

        do {
            fileDescriptor = shm_open(tempName.data(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
        } while (fileDescriptor == -1 && errno == EINTR);
    }

    if (fileDescriptor != -1)
        shm_unlink(tempName.data());
#endif

    return UnixFileDescriptor { fileDescriptor, UnixFileDescriptor::Adopt };
}

RefPtr<SharedMemory> SharedMemory::allocate(size_t size)
{
    auto fileDescriptor = createSharedMemory();
    if (!fileDescriptor) {
        WTFLogAlways("Failed to create shared memory: %s", safeStrerror(errno).data());
        return nullptr;
    }

    while (ftruncate(fileDescriptor.value(), size) == -1) {
        if (errno != EINTR)
            return nullptr;
    }

    void* data = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fileDescriptor.value(), 0);
    if (data == MAP_FAILED)
        return nullptr;

    RefPtr<SharedMemory> instance = adoptRef(new SharedMemory());
    instance->m_data = data;
    instance->m_fileDescriptor = WTFMove(fileDescriptor);
    instance->m_size = size;
    return instance;
}

RefPtr<SharedMemory> SharedMemory::map(Handle&& handle, Protection protection)
{
    void* data = mmap(0, handle.size(), accessModeMMap(protection), MAP_SHARED, handle.m_handle.value(), 0);
    if (data == MAP_FAILED)
        return nullptr;

    RefPtr<SharedMemory> instance = adoptRef(new SharedMemory());
    instance->m_data = data;
    instance->m_size = handle.size();
    return instance;
}

RefPtr<SharedMemory> SharedMemory::wrapMap(void* data, size_t size, int fileDescriptor)
{
    RefPtr<SharedMemory> instance = adoptRef(new SharedMemory());
    instance->m_data = data;
    instance->m_size = size;
    instance->m_fileDescriptor = UnixFileDescriptor { fileDescriptor, UnixFileDescriptor::Adopt };
    instance->m_isWrappingMap = true;
    return instance;
}

SharedMemory::~SharedMemory()
{
    if (m_isWrappingMap) {
        auto wrapped = m_fileDescriptor.release();
        UNUSED_VARIABLE(wrapped);
        return;
    }

    munmap(m_data, m_size);
}

auto SharedMemory::createHandle(Protection) -> std::optional<Handle>
{
    // FIXME: Handle the case where the passed Protection is ReadOnly.
    // See https://bugs.webkit.org/show_bug.cgi?id=131542.

    UnixFileDescriptor duplicate { m_fileDescriptor.value(), UnixFileDescriptor::Duplicate };
    if (!duplicate) {
        ASSERT_NOT_REACHED();
        return std::nullopt;
    }
    return { Handle(WTFMove(duplicate), m_size) };
}

} // namespace WebCore

#endif
