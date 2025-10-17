/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 14, 2025.
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

#include <limits.h>
#include <wtf/FastMalloc.h>
#include <wtf/Locker.h>
#include <wtf/Noncopyable.h>
#include <wtf/WallTime.h>

#if OS(WINDOWS)
#include <windows.h>
#endif

#if USE(PTHREADS)
#include <pthread.h>
#if !defined(PTHREAD_KEYS_MAX)
// PTHREAD_KEYS_MAX is not defined in bionic nor in Hurd, so explicitly define it here.
#define PTHREAD_KEYS_MAX 1024
#endif
#endif

namespace WTF {

using ThreadFunction = void (*)(void* argument);

#if USE(PTHREADS)
using PlatformThreadHandle = pthread_t;
using PlatformMutex = pthread_mutex_t;
using PlatformCondition = pthread_cond_t;
using ThreadSpecificKey = pthread_key_t;
#if OS(LINUX)
using ThreadIdentifier = pid_t;
#endif
#elif OS(WINDOWS)
using ThreadIdentifier = uint32_t;
using PlatformThreadHandle = HANDLE;
using PlatformMutex = SRWLOCK;
using PlatformCondition = CONDITION_VARIABLE;
using ThreadSpecificKey = DWORD;
#else
#error "Not supported platform"
#endif

class Mutex final {
    WTF_MAKE_NONCOPYABLE(Mutex);
    WTF_MAKE_FAST_ALLOCATED;
public:
    constexpr Mutex() = default;
    WTF_EXPORT_PRIVATE ~Mutex();

    WTF_EXPORT_PRIVATE void lock();
    WTF_EXPORT_PRIVATE bool tryLock();
    WTF_EXPORT_PRIVATE void unlock();

    PlatformMutex& impl() { return m_mutex; }

private:
#if USE(PTHREADS)
    PlatformMutex m_mutex = PTHREAD_MUTEX_INITIALIZER;
#elif OS(WINDOWS)
    PlatformMutex m_mutex = SRWLOCK_INIT;
#endif
};

typedef Locker<Mutex> MutexLocker;

class ThreadCondition final {
    WTF_MAKE_NONCOPYABLE(ThreadCondition);
    WTF_MAKE_FAST_ALLOCATED;
public:
    constexpr ThreadCondition() = default;
    WTF_EXPORT_PRIVATE ~ThreadCondition();
    
    WTF_EXPORT_PRIVATE void wait(Mutex& mutex);
    // Returns true if the condition was signaled before absoluteTime, false if the absoluteTime was reached or is in the past.
    WTF_EXPORT_PRIVATE bool timedWait(Mutex&, WallTime absoluteTime);
    WTF_EXPORT_PRIVATE void signal();
    WTF_EXPORT_PRIVATE void broadcast();
    
private:
#if USE(PTHREADS)
    PlatformCondition m_condition = PTHREAD_COND_INITIALIZER;
#elif OS(WINDOWS)
    PlatformCondition m_condition = CONDITION_VARIABLE_INIT;
#endif
};

#if USE(PTHREADS)

static constexpr ThreadSpecificKey InvalidThreadSpecificKey = PTHREAD_KEYS_MAX;

inline void threadSpecificKeyCreate(ThreadSpecificKey* key, void (*destructor)(void *))
{
    int error = pthread_key_create(key, destructor);
    if (error)
        CRASH();
}

inline void threadSpecificKeyDelete(ThreadSpecificKey key)
{
    int error = pthread_key_delete(key);
    if (error)
        CRASH();
}

inline void threadSpecificSet(ThreadSpecificKey key, void* value)
{
    pthread_setspecific(key, value);
}

inline void* threadSpecificGet(ThreadSpecificKey key)
{
    return pthread_getspecific(key);
}

#endif

} // namespace WTF

using WTF::Mutex;
using WTF::MutexLocker;
using WTF::ThreadCondition;
