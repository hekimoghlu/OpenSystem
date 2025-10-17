/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 7, 2022.
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

#include <wtf/ArgumentCoder.h>
#include <wtf/Forward.h>
#include <wtf/ThreadSafeRefCounted.h>

#if USE(UNIX_DOMAIN_SOCKETS)
#include <wtf/unix/UnixFileDescriptor.h>
#endif

#if OS(WINDOWS)
#include <wtf/win/Win32Handle.h>
#endif

#if OS(DARWIN)
#include <wtf/MachSendRight.h>
#endif

#if PLATFORM(COCOA)
OBJC_CLASS NSData;
#endif

namespace WebCore {

class FragmentedSharedBuffer;
class ProcessIdentity;
class SharedBuffer;

enum class MemoryLedger { None, Default, Network, Media, Graphics, Neural };

WEBCORE_EXPORT bool isMemoryAttributionDisabled();

class SharedMemoryHandle {
public:
    using Type =
#if USE(UNIX_DOMAIN_SOCKETS)
        UnixFileDescriptor;
#elif OS(DARWIN)
        MachSendRight;
#elif OS(WINDOWS)
        Win32Handle;
#endif

    SharedMemoryHandle(SharedMemoryHandle&&) = default;
#if USE(UNIX_DOMAIN_SOCKETS)
    explicit SharedMemoryHandle(const SharedMemoryHandle&);
#else
    explicit SharedMemoryHandle(const SharedMemoryHandle&) = default;
#endif
    WEBCORE_EXPORT SharedMemoryHandle(SharedMemoryHandle::Type&&, size_t);

    SharedMemoryHandle& operator=(SharedMemoryHandle&&) = default;

    size_t size() const { return m_size; }

    // Take ownership of the memory for process memory accounting purposes.
    WEBCORE_EXPORT void takeOwnershipOfMemory(MemoryLedger) const;
    // Transfer ownership of the memory for process memory accounting purposes.
    WEBCORE_EXPORT void setOwnershipOfMemory(const WebCore::ProcessIdentity&, MemoryLedger) const;

#if USE(UNIX_DOMAIN_SOCKETS)
    UnixFileDescriptor releaseHandle();
#endif

private:
    friend struct IPC::ArgumentCoder<SharedMemoryHandle, void>;
    friend class SharedMemory;

    Type m_handle;
    size_t m_size { 0 };
};

class SharedMemory : public ThreadSafeRefCounted<SharedMemory> {
public:
    using Handle = SharedMemoryHandle;

    enum class Protection : bool { ReadOnly, ReadWrite };

    // FIXME: Change these factory functions to return Ref<SharedMemory> and crash on failure.
    WEBCORE_EXPORT static RefPtr<SharedMemory> allocate(size_t);
    WEBCORE_EXPORT static RefPtr<SharedMemory> copyBuffer(const WebCore::FragmentedSharedBuffer&);
    WEBCORE_EXPORT static RefPtr<SharedMemory> copySpan(std::span<const uint8_t>);
    WEBCORE_EXPORT static RefPtr<SharedMemory> map(Handle&&, Protection);
#if USE(UNIX_DOMAIN_SOCKETS)
    WEBCORE_EXPORT static RefPtr<SharedMemory> wrapMap(void*, size_t, int fileDescriptor);
#elif OS(DARWIN)
    WEBCORE_EXPORT static RefPtr<SharedMemory> wrapMap(std::span<const uint8_t>, Protection);
#endif

    WEBCORE_EXPORT ~SharedMemory();

    WEBCORE_EXPORT std::optional<Handle> createHandle(Protection);

    size_t size() const { return m_size; }

    std::span<const uint8_t> span() const { return unsafeMakeSpan(static_cast<const uint8_t*>(m_data), m_size); }
    std::span<uint8_t> mutableSpan() const { return unsafeMakeSpan(static_cast<uint8_t*>(m_data), m_size); }

#if OS(WINDOWS)
    HANDLE handle() const { return m_handle.get(); }
#endif

#if PLATFORM(COCOA)
    Protection protection() const { return m_protection; }
    WEBCORE_EXPORT RetainPtr<NSData> toNSData() const;
#endif

    WEBCORE_EXPORT Ref<WebCore::SharedBuffer> createSharedBuffer(size_t) const;

private:
#if OS(DARWIN)
    MachSendRight createSendRight(Protection) const;
#endif

    size_t m_size;
    void* m_data;
#if PLATFORM(COCOA)
    Protection m_protection;
#endif

#if USE(UNIX_DOMAIN_SOCKETS)
    UnixFileDescriptor m_fileDescriptor;
    bool m_isWrappingMap { false };
#elif OS(DARWIN)
    MachSendRight m_sendRight;
#elif OS(WINDOWS)
    Win32Handle m_handle;
#endif
};

} // namespace WebCore
