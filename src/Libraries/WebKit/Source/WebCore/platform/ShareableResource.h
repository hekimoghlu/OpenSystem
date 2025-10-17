/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 24, 2025.
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

#if ENABLE(SHAREABLE_RESOURCE)

#include "SharedMemory.h"
#include <wtf/ArgumentCoder.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>

namespace WebCore {

class SharedBuffer;

class ShareableResourceHandle {
    WTF_MAKE_NONCOPYABLE(ShareableResourceHandle);
public:
    ShareableResourceHandle(ShareableResourceHandle&&) = default;
    WEBCORE_EXPORT ShareableResourceHandle(SharedMemory::Handle&&, unsigned, unsigned);

    ShareableResourceHandle& operator=(ShareableResourceHandle&&) = default;

    unsigned size() const { return m_size; }

    WEBCORE_EXPORT RefPtr<SharedBuffer> tryWrapInSharedBuffer() &&;

private:
    friend struct IPC::ArgumentCoder<ShareableResourceHandle, void>;
    friend class ShareableResource;

    SharedMemory::Handle m_handle;
    unsigned m_offset { 0 };
    unsigned m_size { 0 };
};

class ShareableResource : public ThreadSafeRefCounted<ShareableResource> {
public:
    using Handle = ShareableResourceHandle;

    // Create a shareable resource that uses malloced memory.
    WEBCORE_EXPORT static RefPtr<ShareableResource> create(Ref<SharedMemory>&&, unsigned offset, unsigned size);

    // Create a shareable resource from a handle.
    static RefPtr<ShareableResource> map(Handle&&);

    WEBCORE_EXPORT std::optional<Handle> createHandle();

    WEBCORE_EXPORT ~ShareableResource();

    unsigned size() const;
    std::span<const uint8_t> span() const;

private:
    friend class ShareableResourceHandle;

    ShareableResource(Ref<SharedMemory>&&, unsigned offset, unsigned size);
    RefPtr<SharedBuffer> wrapInSharedBuffer();

    Ref<SharedMemory> m_sharedMemory;

    const unsigned m_offset;
    const unsigned m_size;
};

} // namespace WebCore

#endif // ENABLE(SHAREABLE_RESOURCE)
