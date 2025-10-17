/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 16, 2025.
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
#ifndef GMutexLocker_h
#define GMutexLocker_h

#if USE(GLIB)

#include <glib.h>
#include <wtf/Noncopyable.h>

namespace WTF {

template<typename T>
struct MutexWrapper;

template<>
struct MutexWrapper<GMutex> {
    static void lock(GMutex* mutex)
    {
        g_mutex_lock(mutex);
    }

    static void unlock(GMutex* mutex)
    {
        g_mutex_unlock(mutex);
    }
};

template<>
struct MutexWrapper<GRecMutex> {
    static void lock(GRecMutex* mutex)
    {
        g_rec_mutex_lock(mutex);
    }

    static void unlock(GRecMutex* mutex)
    {
        g_rec_mutex_unlock(mutex);
    }
};

template<typename T>
class GMutexLocker {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_MAKE_NONCOPYABLE(GMutexLocker);
public:
    explicit GMutexLocker(T& mutex)
        : m_mutex(mutex)
        , m_locked(false)
    {
        lock();
    }

    ~GMutexLocker()
    {
        unlock();
    }

    void lock()
    {
        if (m_locked)
            return;

        MutexWrapper<T>::lock(&m_mutex);
        m_locked = true;
    }

    void unlock()
    {
        if (!m_locked)
            return;

        m_locked = false;
        MutexWrapper<T>::unlock(&m_mutex);
    }

private:
    T& m_mutex;
    bool m_locked;
};

} // namespace WTF

#endif // USE(GLIB)

#endif // GMutexLocker_h
