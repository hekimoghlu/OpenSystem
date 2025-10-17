/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 2, 2024.
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

#include "Timeout.h"
#include <wtf/Lock.h>
#include <wtf/Noncopyable.h>
#include <wtf/Seconds.h>
#include <wtf/TZoneMallocInlines.h>

#if PLATFORM(COCOA)
#include <mach/semaphore.h>
#include <wtf/MachSendRight.h>
#elif OS(WINDOWS)
#include <wtf/win/Win32Handle.h>
#elif USE(UNIX_DOMAIN_SOCKETS)
#include <wtf/unix/UnixFileDescriptor.h>
#endif

namespace IPC {

class Decoder;
class Encoder;
struct EventSignalPair;

std::optional<EventSignalPair> createEventSignalPair();

// A semaphore implementation that can be duplicated across IPC.
// The cocoa implementation of this only interrupts wait calls upon the
// remote process terminating if the Semaphore was created by the remote
// process.
// It is generally preferred to start using IPC::Event/Signal instead
// to avoid this.
class Semaphore {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(Semaphore);
    WTF_MAKE_NONCOPYABLE(Semaphore);
public:
    Semaphore();
    Semaphore(Semaphore&&);
    ~Semaphore();
    Semaphore& operator=(Semaphore&&);

    void signal();
    bool wait();
    bool waitFor(Timeout);

#if PLATFORM(COCOA)
    explicit Semaphore(MachSendRight&&);

    MachSendRight createSendRight() const;
    explicit operator bool() const { return m_sendRight || m_semaphore != SEMAPHORE_NULL; }
#elif OS(WINDOWS)
    explicit Semaphore(Win32Handle&&);
    Win32Handle win32Handle() const { return Win32Handle { m_semaphoreHandle }; }

#elif USE(UNIX_DOMAIN_SOCKETS)
    explicit Semaphore(UnixFileDescriptor&&);
    UnixFileDescriptor duplicateDescriptor() const;
    explicit operator bool() const { return !!m_fd; }
#else
    explicit operator bool() const { return true; }
#endif

private:
    friend std::optional<EventSignalPair> createEventSignalPair();
    void destroy();
#if PLATFORM(COCOA)
    MachSendRight m_sendRight;
    semaphore_t m_semaphore { SEMAPHORE_NULL };
#elif OS(WINDOWS)
    Win32Handle m_semaphoreHandle;
#elif USE(UNIX_DOMAIN_SOCKETS)
    UnixFileDescriptor m_fd;
#endif
};

} // namespace IPC
