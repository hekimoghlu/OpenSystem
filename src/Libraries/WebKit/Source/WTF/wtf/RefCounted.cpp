/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 2, 2022.
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
#include <wtf/RefCounted.h>
#include <wtf/StackShot.h>

#include <wtf/ThreadSafeRefCounted.h>

namespace WTF {

bool RefCountedBase::areThreadingChecksEnabledGlobally { false };

static_assert(sizeof(RefCountedBase) == sizeof(ThreadSafeRefCountedBase));

class RefLogStackShot : public StackShot {
    static const size_t s_size = 18;
    static const size_t s_skip = 6;
public:
    RefLogStackShot(const void* ptr)
        : StackShot(s_size)
        , m_ptr(ptr)
    {
    }

    const void* ptr() { return m_ptr; }

    void print()
    {
        if (size() < s_skip)
            return;
        WTFPrintBacktrace(span().subspan(s_skip));
    }

private:
    const void* m_ptr;
};

class RefLogSingleton {
public:
    static void append(const void* ptr)
    {
        RefLogStackShot* stackShot = new RefLogStackShot(ptr);

        size_t index = s_end.fetch_add(1, std::memory_order_acquire) & s_sizeMask;
        if (RefLogStackShot* old = s_buffer[index].exchange(nullptr, std::memory_order_acquire))
            delete old;

        // Other threads may have raced ahead and filled the log. If so, our stack shot is oldest, so we drop it.
        RefLogStackShot* expected = nullptr;
        if (!s_buffer[index].compare_exchange_strong(expected, stackShot, std::memory_order_release))
            delete stackShot;
    }

    template<typename Callback>
    static void forEachLIFO(const Callback& callback)
    {
        size_t last = s_end.load(std::memory_order_acquire) - 1;
        for (size_t i = 0; i < s_size; ++i) {
            size_t index = (last - i) & s_sizeMask;
            RefLogStackShot* stackShot = s_buffer[index].load(std::memory_order_acquire);
            if (!stackShot)
                continue;
            callback(stackShot);
        }
    }

private:
    static constexpr size_t s_size = 512; // Keep the log short to decrease the odds of logging a previous alias at the same address.
    static constexpr size_t s_sizeMask = s_size - 1;

    static std::atomic<size_t> s_end;
    static std::array<std::atomic<RefLogStackShot*>, s_size> s_buffer;
};

std::atomic<size_t> RefLogSingleton::s_end;
std::array<std::atomic<RefLogStackShot*>, RefLogSingleton::s_size> RefLogSingleton::s_buffer;

void RefCountedBase::logRefDuringDestruction(const void* ptr)
{
    RefLogSingleton::append(ptr);
}

void RefCountedBase::printRefDuringDestructionLogAndCrash(const void* ptr)
{
    WTFLogAlways("Error: Dangling RefPtr: %p", ptr);
    WTFLogAlways("This means that a ref() during destruction was not balanced by a deref() before destruction ended.");
    WTFLogAlways("Below are the most recent ref()'s during destruction for this address.");

    RefLogSingleton::forEachLIFO([&] (RefLogStackShot* stackShot) {
        if (stackShot->ptr() != ptr)
            return;

        WTFLogAlways(" ");

        stackShot->print();
    });

    CRASH_WITH_SECURITY_IMPLICATION();
}

}
