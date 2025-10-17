/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 29, 2023.
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

#include "Decoder.h"
#include "Encoder.h"
#include "IPCSemaphore.h"
#include <wtf/TZoneMallocInlines.h>

namespace IPC {

struct EventSignalPair;
std::optional<EventSignalPair> createEventSignalPair();

// A waitable Event object (and corresponding Signal object), where the
// Signal object can be serialized across IPC.
// The wait operation will be interrupted if the Signal instance is
// destroyed (including if the remote process that owns it crashes or is
// killed).
// FIXME: Write proper interruptible implementations for non-COCOA platforms.
class Signal {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(Signal);
    WTF_MAKE_NONCOPYABLE(Signal);
public:
    Signal(Signal&& other) = default;
    Signal& operator=(Signal&& other) = default;

#if PLATFORM(COCOA)
    void signal();

    MachSendRight takeSendRight() { return WTFMove(m_sendRight); }
#else
    void signal()
    {
        m_semaphore.signal();
    }

    const Semaphore& semaphore() const { return m_semaphore; }
#endif

private:
    friend struct IPC::ArgumentCoder<Signal, void>;

    friend std::optional<EventSignalPair> createEventSignalPair();

#if PLATFORM(COCOA)
    Signal(MachSendRight&& sendRight)
        : m_sendRight(WTFMove(sendRight))
    { }
#else
    Signal(Semaphore&& semaphore)
        : m_semaphore(WTFMove(semaphore))
    { }
#endif

#if PLATFORM(COCOA)
    MachSendRight m_sendRight;
#else
    Semaphore m_semaphore;
#endif
};

class Event {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(Event);
    WTF_MAKE_NONCOPYABLE(Event);
public:
#if PLATFORM(COCOA)
    ~Event();
    Event(Event&& other)
        : m_receiveRight(WTFMove(other.m_receiveRight))
    {
        other.m_receiveRight = MACH_PORT_NULL;
    }

    Event& operator=(Event&& other)
    {
        m_receiveRight = WTFMove(other.m_receiveRight);
        other.m_receiveRight = MACH_PORT_NULL;
        return *this;
    }

    bool wait();
    bool waitFor(Timeout);
#else
    Event(Event&& other) = default;
    Event& operator=(Event&& other) = default;

    bool wait()
    {
        return m_semaphore.wait();
    }

    bool waitFor(Timeout timeout)
    {
        return m_semaphore.waitFor(timeout);
    }
#endif

private:
    friend std::optional<EventSignalPair> createEventSignalPair();

#if PLATFORM(COCOA)
    Event(mach_port_t port)
        : m_receiveRight(port)
    { }
#else
    Event(Semaphore&& semaphore)
        : m_semaphore(WTFMove(semaphore))
    { }
#endif

#if PLATFORM(COCOA)
    mach_port_t m_receiveRight;
#else
    Semaphore m_semaphore;
#endif
};

struct EventSignalPair {
    Event event;
    Signal signal;
};

#if !PLATFORM(COCOA)
inline std::optional<EventSignalPair> createEventSignalPair()
{
    Semaphore event;
#if PLATFORM(WIN)
    Semaphore signal(Win32Handle { event.m_semaphoreHandle });
#else
    Semaphore signal(event.m_fd.duplicate());
#endif
    return EventSignalPair { Event { WTFMove(event) }, Signal { WTFMove(signal) } };
}
#endif

} // namespace IPC
