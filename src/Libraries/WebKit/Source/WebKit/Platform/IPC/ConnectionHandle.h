/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 4, 2023.
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

#if USE(UNIX_DOMAIN_SOCKETS)
#include <wtf/unix/UnixFileDescriptor.h>
#elif OS(DARWIN)
#include <wtf/MachSendRight.h>
#elif OS(WINDOWS)
#include <wtf/win/Win32Handle.h>
#endif

namespace IPC {

class ConnectionHandle {
public:
    ConnectionHandle() = default;
    ConnectionHandle(ConnectionHandle&&) = default;
    ConnectionHandle& operator=(ConnectionHandle&&) = default;

    ConnectionHandle(const ConnectionHandle&) = delete;
    ConnectionHandle& operator=(const ConnectionHandle&) = delete;

#if USE(UNIX_DOMAIN_SOCKETS)
    ConnectionHandle(UnixFileDescriptor&& inHandle)
        : m_handle(WTFMove(inHandle))
    { }
    explicit operator bool() const { return !!m_handle; }
    int release() WARN_UNUSED_RETURN { return m_handle.release(); }
#elif OS(WINDOWS)
    ConnectionHandle(Win32Handle&& inHandle)
        : m_handle(WTFMove(inHandle))
    { }
    explicit operator bool() const { return !!m_handle; }
    HANDLE leak() WARN_UNUSED_RETURN { return m_handle.leak(); }
#elif OS(DARWIN)
    ConnectionHandle(MachSendRight&& sendRight)
        : m_handle(WTFMove(sendRight))
    { }
    explicit operator bool() const { return MACH_PORT_VALID(m_handle.sendRight()); }
    mach_port_t leakSendRight() WARN_UNUSED_RETURN { return m_handle.leakSendRight(); }
#endif

private:
    friend struct IPC::ArgumentCoder<ConnectionHandle, void>;

#if USE(UNIX_DOMAIN_SOCKETS)
    UnixFileDescriptor m_handle;
#elif OS(WINDOWS)
    Win32Handle m_handle;
#elif OS(DARWIN)
    MachSendRight m_handle;
#endif
};

} // namespace IPC
