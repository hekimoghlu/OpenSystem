/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 20, 2024.
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
#include "WPEWaylandSHMPool.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <wtf/CryptographicallyRandomNumber.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/CString.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/WTFString.h>

#if HAVE(LINUX_MEMFD_H)
#include <linux/memfd.h>
#include <sys/syscall.h>
#endif

namespace WPE {

WTF_MAKE_TZONE_ALLOCATED_IMPL(WaylandSHMPool);

static UnixFileDescriptor createSharedMemory()
{
    int fileDescriptor = -1;

#if HAVE(LINUX_MEMFD_H)
    static bool isMemFdAvailable = true;
    if (isMemFdAvailable) {
        do {
            fileDescriptor = syscall(__NR_memfd_create, "WPEWaylandSHMPool", MFD_CLOEXEC);
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
        auto name = makeString("/WPEWaylandSHMPool."_s, cryptographicallyRandomNumber<unsigned>());
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

std::unique_ptr<WaylandSHMPool> WaylandSHMPool::create(struct wl_shm* shm, size_t size)
{
    auto fd = createSharedMemory();
    if (!fd)
        return nullptr;

    while (ftruncate(fd.value(), size) == -1) {
        if (errno != EINTR)
            return nullptr;
    }

    void* data = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd.value(), 0);
    if (data == MAP_FAILED)
        return nullptr;

    return makeUnique<WaylandSHMPool>(data, size, WTFMove(fd), shm);
}

WaylandSHMPool::WaylandSHMPool(void* data, size_t size, UnixFileDescriptor&& fd, struct wl_shm* shm)
    : m_data(data)
    , m_size(size)
    , m_fd(WTFMove(fd))
    , m_pool(wl_shm_create_pool(shm, m_fd.value(), m_size))
{
}

WaylandSHMPool::~WaylandSHMPool()
{
    wl_shm_pool_destroy(m_pool);
    if (m_data != MAP_FAILED)
        munmap(m_data, m_size);
}

int WaylandSHMPool::allocate(size_t size)
{
    if (m_used + size > m_size) {
        if (!resize(2 * m_size + size))
            return -1;
    }

    int offset = m_used;
    m_used += size;
    return offset;
}

bool WaylandSHMPool::resize(size_t size)
{
    while (ftruncate(m_fd.value(), size) == -1) {
        if (errno != EINTR)
            return false;
    }

    wl_shm_pool_resize(m_pool, size);

    if (m_data != MAP_FAILED)
        munmap(m_data, m_size);
    m_data = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, m_fd.value(), 0);
    if (m_data == MAP_FAILED)
        return false;

    m_size = size;
    return true;
}

struct wl_buffer* WaylandSHMPool::createBuffer(uint32_t offset, uint32_t width, uint32_t height, uint32_t stride)
{
    return wl_shm_pool_create_buffer(m_pool, offset, width, height, stride, WL_SHM_FORMAT_ARGB8888);
}

} // namespace WPE
