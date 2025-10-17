/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 2, 2023.
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
* c11threads.h
* C11 threads support package
**********************************************************************/

#ifndef _OBJC_C11THREADS_H
#define _OBJC_C11THREADS_H

#include <threads.h>

#include "objc-config.h"

// .. Basics ...........................................................

#define OBJC_THREAD_T_DEFINED 1
typedef thrd_t objc_thread_t;

static inline int objc_thread_equal(objc_thread_t t1, objc_thread_t t2) {
    return thrd_equal(t1, t2);
}

static inline bool objc_is_threaded() {
    return true;
}

static inline objc_thread_t objc_thread_self()
{
    return thrd_current();
}

// .. objc_tls .........................................................

template <class T, typename Destructor>
class objc_tls_base {
public:
    using type = T;

    static_assert(sizeof(T) <= sizeof(void *), "T must fit in a void *");

private:
    tss_t key_;

    static void dtor_(void *ptr) {
        Destructor d;
        d((T)(uintptr_t)ptr);
    }
protected:
    objc_tls_base() { tss_create(&key_, dtor_); }
    ~objc_tls_base() { tss_delete(key_); }

    ALWAYS_INLINE T get_() const { return (T)(uintptr_t)tss_get(key_); }
    ALWAYS_INLINE void set_(T newval) {
        tss_set(key_, (void *)(uintptr_t)newval);
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
    tss_t key_;

protected:
    objc_tls_base() { tss_create(&key_, nullptr); }
    ~objc_tls_base() { tss_delete(key_); }

    ALWAYS_INLINE T get_() const { return (T)(uintptr_t)tss_get(key_); }
    ALWAYS_INLINE void set_(T newval) {
        tss_set(key_, (void *)(uintptr_t)newval);
    }
};

// .. tls_autoptr ......................................................

template <class T>
class tls_autoptr_impl {
private:
    tss_t key_;

    static void dtor_(void *ptr) {
        delete (T *)ptr;
    }

    ALWAYS_INLINE T *get_(bool create) const {
        T *ptr = (T *)tss_get(key_);
        if (create && !ptr) {
            ptr = new T();
            tss_set(key_, ptr);
        }
        return ptr;
    }
    ALWAYS_INLINE void set_(T* newptr) {
        T *ptr = (T *)tss_get(key_);
        if (ptr)
            delete ptr;
        tss_set(key_, newptr);
    }

public:
    tls_autoptr_impl() { tss_create(&key_, dtor_); }
    ~tls_autoptr_impl() { tss_delete(key_); }

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

class objc_lock_base_t : nocopy_t {
    mtx_t lock_;
    void init() {
        mtx_init(&lock_, mtx_plain);
    }
public:
    objc_lock_base_t() {
        init();
    }
    ~objc_lock_base_t() {
        mtx_destroy(&lock_);
    }

    void lock() {
        mtx_lock(&lock_);
    }

    bool tryLock() {
        return mtx_trylock(&lock_) == thrd_success;
    }

    void unlock() {
        mtx_unlock(&lock_);
    }

    bool tryUnlock() {
        return mtx_unlock(&lock_) == thrd_success;
    }

    void reset() {
        memset(&lock_, 0, sizeof(lock_));
        init();
    }
};

// .. objc_recursive_lock_t ............................................

class objc_recursive_lock_base_t : nocopy_t {
    mtx_t lock_;

    void init() {
        mtx_init(&lock_, mtx_recursive);
    }
public:
    objc_recursive_lock_base_t() {
        init();
    }
    ~objc_recursive_lock_base_t() {
        mtx_destroy(&lock_);
    }

    void lock() {
        mtx_lock(&lock_);
    }

    bool tryLock() {
        return mtx_trylock(&lock_) == thrd_success;
    }

    void unlock() {
        mtx_unlock(&lock_);
    }

    bool tryUnlock() {
        return mtx_unlock(&lock_) == thrd_success;
    }

    void unlockForkedChild() {
        unlock();
    }

    void reset() {
        memset(&lock_, 0, sizeof(lock_));
        init();
    }

    void hardReset() {
        reset();
    }
};

#endif // _OBJC_C11THREADS_H
