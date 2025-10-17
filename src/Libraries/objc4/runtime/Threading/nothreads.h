/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 3, 2022.
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
* nothreads.h
* Single threading support package
**********************************************************************/

#ifndef _OBJC_NOTHREADS_H
#define _OBJC_NOTHREADS_H

#include "objc-config.h"

// .. Direct thread keys ...............................................

#define SUPPORT_DIRECT_THREAD_KEYS 1

// .. Basics ...........................................................

typedef struct objc_thread *objc_thread_t;

static inline int objc_thread_equal(objc_thread_t t1, objc_thread_t t2) {
    return t1 == t2;
}

static inline bool objc_is_threaded() {
    return false;
}

static inline objc_thread_t objc_thread_self()
{
    return nullptr;
}

// .. objc_tls .........................................................

template <class T, typename Destructor>
class objc_tls_base {
public:
    using type = T;

    static_assert(sizeof(T) <= sizeof(void *), "T must fit in a void *");

private:
    T value_;

protected:
    objc_tls_base() {}

    ALWAYS_INLINE T get_() const { return value_; }
    ALWAYS_INLINE void set_(T newval) { value_ = newval; }
};

// .. tls_autoptr ......................................................

template <class T>
class tls_autoptr_impl {
private:
    T *ptr_;

    ALWAYS_INLINE T *get_(bool create) const {
        if (create && !ptr_)
            ptr_ = new T();
        return ptr_;
    }
    ALWAYS_INLINE void set_(T *newptr) {
        if (ptr_)
            delete ptr_;
        ptr_ = newptr;
    }

public:
    tls_autoptr_impl() {}
    ~tls_autoptr_impl() { if (ptr_) delete ptr_; }

    ALWAYS_INLINE tls_autoptr& operator=(T *newptr) {
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
public:
    objc_lock_base_t() {}

    void lock() {}
    bool tryLock() { return true; }
    void unlock() {}
    bool tryUnlock() { return true; }
    void reset() {}
};

// .. objc_recursive_lock_t ............................................

class objc_recursive_lock_base_t : nocopy_t {
public:
    objc_recursive_lock_base_t() {}

    void lock() {}
    bool tryLock() { return true; }
    void unlock() {}
    bool tryUnlock() { return true; }
    void unlockForkedChild() {}
    void reset() {}
    void hardReset() {}
};

#endif // _OBJC_NOTHREADS_H
