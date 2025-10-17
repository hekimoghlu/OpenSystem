/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 20, 2023.
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
#pragma once

#include <wayland-client.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/unix/UnixFileDescriptor.h>

namespace WPE {

class WaylandSHMPool {
    WTF_MAKE_TZONE_ALLOCATED(WaylandSHMPool);
public:
    static std::unique_ptr<WaylandSHMPool> create(struct wl_shm*, size_t);

    WaylandSHMPool(void*, size_t, WTF::UnixFileDescriptor&&, struct wl_shm*);
    ~WaylandSHMPool();

    void* data() const { return m_data; }
    size_t size() const { return m_size; }

    int allocate(size_t);
    struct wl_buffer* createBuffer(uint32_t offset, uint32_t width, uint32_t height, uint32_t stride);

private:
    bool resize(size_t);

    void* m_data { nullptr };
    size_t m_size { 0 };
    WTF::UnixFileDescriptor m_fd;
    struct wl_shm_pool* m_pool { nullptr };
    size_t m_used { 0 };
};

} // namespace WPE
