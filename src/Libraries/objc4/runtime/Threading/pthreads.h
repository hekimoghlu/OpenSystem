/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 18, 2025.
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
/***********************************************************************
* pthread.h
* Pthread support package
**********************************************************************/

#ifndef _OBJC_PTHREAD_H
#define _OBJC_PTHREAD_H

#include <pthread.h>

#include "objc-config.h"

// .. Basics ...........................................................

#define OBJC_THREAD_T_DEFINED 1
typedef pthread_t objc_thread_t;

static inline int objc_thread_equal(objc_thread_t t1, objc_thread_t t2) {
    return pthread_equal(t1, t2);
}

#if !_OBJC_PTHREAD_IS_DARWIN
static inline bool objc_is_threaded() {
    return true;
}

__attribute__((const))
static inline objc_thread_t objc_thread_self()
{
    return pthread_self();
}
#endif

// .. objc_tls .........................................................

template <class T, typename Destructor>
class objc_tls_base {
public:
    using type = T;

    static_assert(sizeof(T) <= sizeof(void *), "T must fit in a void *");

private:
    pthread_key_t key_;

    static void dtor_(void *ptr) {
        Destructor d;
        d((T)(uintptr_t)ptr);
    }
protected:
    objc_tls_base() { pthread_key_create(&key_, dtor_); }
    ~objc_tls_base() { pthread_key_delete(key_); }

    ALWAYS_INLINE T get_() const {
        return (T)(uintptr_t)pthread_getspecific(key_);
    }
    ALWAYS_INLINE void set_(T newval) {
        pthread_setspecific(key_, (void *)(uintptr_t)newval);
    }
};

template <class T>
class objc_tls_base<T, void> {
public:
    using type = T;

    static_assert(sizeof(T) <= sizeof(void *), "T must fit in a void *");
    static_assert(std::is_trivially_destructible<T>::value,
                  "T must be trivially destructible");

private:
    pthread_key_t key_;

protected:
    objc_tls_base() { pthread_key_create(&key_, nullptr); }
    ~objc_tls_base() { pthread_key_delete(key_); }

    ALWAYS_INLINE T get_() const {
        return (T)(uintptr_t)pthread_getspecific(key_);
    }
    ALWAYS_INLINE void set_(T newval) {
        pthread_setspecific(key_, (void *)(uintptr_t)newval);
    }
};

// .. tls_autoptr ......................................................

template <class T>
class tls_autoptr_impl {
private:
    pthread_key_t key_;

    static void dtor_(void *ptr) {
        delete (T *)ptr;
    }

    ALWAYS_INLINE T *get_(bool create) const {
        T *ptr = (T *)pthread_getspecific(key_);
        if (create && !ptr) {
            ptr = new T();
            pthread_setspecific(key_, ptr);
        }
        return ptr;
    }
    ALWAYS_INLINE void set_(T* newptr) {
        T *ptr = (T *)pthread_getspecific(key_);
        if (ptr)
            delete ptr;
        pthread_setspecific(key_, newptr);
    }

public:
    tls_autoptr_impl() { pthread_key_create(&key_, dtor_); }
    ~tls_autoptr_impl() { pthread_key_delete(key_); }

    ALWAYS_INLINE tls_autoptr_impl& operator=(T *newptr) {
        set_(newptr);
        return *this;
    }

    ALWAYS_INLINE T *get(bool create) const { return get_(create); }

    ALWAYS_INLINE operator T*() const {
        return get_(true);
    }

    ALWAYS_INLINE T& operator*() {
        return *get_(true);
    }

    ALWAYS_INLINE T* operator->() {
        return get_(true);
    }
};

// .. objc_lock_t ......................................................

#if !_OBJC_PTHREAD_IS_DARWIN
class objc_lock_base_t : nocopy_t {
    pthread_mutex_t lock_;
public:
    objc_lock_base_t() : lock_(PTHREAD_MUTEX_INITIALIZER) {}
    ~objc_lock_base_t() {
        pthread_mutex_destroy(&lock_);
    }

    void lock() {
        pthread_mutex_lock(&lock_);
    }

    bool tryLock() {
        return pthread_mutex_trylock(&lock_) == 0;
    }

    void unlock() {
        pthread_mutex_unlock(&lock_);
    }

    bool tryUnlock() {
        return pthread_mutex_unlock(&lock_) == 0;
    }

    void reset() {
        memset(&lock_, 0, sizeof(lock_));
        lock_ = PTHREAD_MUTEX_INITIALIZER;
    }

    void hardReset() {
        reset();
    }
};
#endif // !_OBJC_PTHREAD_IS_DARWIN

// .. objc_recursive_lock_t ............................................

#if !_OBJC_PTHREAD_IS_DARWIN
class objc_recursive_lock_base_t : nocopy_t {
    pthread_mutex_t lock_;
private:
    void init() {
        pthread_mutexattr_t attr;
        pthread_mutexattr_init(&attr);
        pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
        pthread_mutex_init(&lock_, &attr);
        pthread_mutexattr_destroy(&attr);
    }
public:
    objc_recursive_lock_base_t() {
        init();
    }
    ~objc_recursive_lock_base_t() {
        pthread_mutex_destroy(&lock_);
    }

    void lock() {
        pthread_mutex_lock(&lock_);
    }

    bool tryLock() {
        return pthread_mutex_trylock(&lock_) == 0;
    }

    void unlock() {
        pthread_mutex_unlock(&lock_);
    }

    bool tryUnlock() {
        return pthread_mutex_unlock(&lock_) == 0;
    }

    void unlockForkedChild() {
        unlock();
    }

    void reset() {
        memset(&lock_, 0, sizeof(lock_));
        init();
    }
};
#endif // !_OBJC_PTHREAD_IS_DARWIN

#endif // _OBJC_PTHREAD_H
