/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 6, 2025.
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
* lockdebug.h
* Lock debugging
**********************************************************************/

#ifndef _OBJC_LOCKDEBUG_H
#define _OBJC_LOCKDEBUG_H

// Define LOCKDEBUG if it isn't already set
#ifndef LOCKDEBUG
#   if DEBUG
#       define LOCKDEBUG 1
#   else
#       define LOCKDEBUG 0
#   endif
#endif

namespace lockdebug {

    // Internal functions
#if LOCKDEBUG
    namespace notify {
        void remember(objc_lock_base_t *lock);
        void lock(objc_lock_base_t *lock);
        void unlock(objc_lock_base_t *lock);

        void remember(objc_recursive_lock_base_t *lock);
        void lock(objc_recursive_lock_base_t *lock);
        void unlock(objc_recursive_lock_base_t *lock);
    }
#endif

    // Use fork_unsafe to get a lock that isn't acquired and released around
    // fork().
    struct fork_unsafe_t {
        constexpr fork_unsafe_t() = default;
    };

    template <class T>
    class lock_mixin: public T {
    public:
#if LOCKDEBUG
        lock_mixin() : T() {
            lockdebug::notify::remember((T *)this);
        }

        lock_mixin(const fork_unsafe_t) : T() {}

        void lock() {
            lockdebug::notify::lock((T *)this);
            T::lock();
        }

        bool tryLock() {
            bool success = T::tryLock();
            if (success)
                lockdebug::notify::lock((T *)this);
            return success;
        }

        void unlock() {
            lockdebug::notify::unlock((T *)this);
            T::unlock();
        }

        bool tryUnlock() {
            bool success = T::tryUnlock();
            if (success)
                lockdebug::notify::unlock((T *)this);
            return success;
        }

        void unlockForkedChild() {
            lockdebug::notify::unlock((T *)this);
            T::unlockForkedChild();
        }

        void reset() {
            lockdebug::notify::unlock((T *)this);
            T::reset();
        }
#else
        lock_mixin() : T() {}
        lock_mixin(const fork_unsafe_t) : T() {}
#endif
    };

    typedef locker_mixin<lockdebug::lock_mixin<objc_lock_base_t>> *(*lock_enumerator)(unsigned);

    // APIs
#if LOCKDEBUG
    void assert_locked(objc_lock_base_t *lock);
    void assert_unlocked(objc_lock_base_t *lock);

    void assert_locked(objc_recursive_lock_base_t *lock);
    void assert_unlocked(objc_recursive_lock_base_t *lock);

    void assert_all_locks_locked();
    void assert_no_locks_locked();

    // Assert that we hold no locks within the global set of known locks, except
    // for the ones given in the list. A list entry may either be a lock pointer
    // or a function that returns a sequence of locks. Functions are called
    // repeatedly with 0, 1, 2, 3, etc. until they return nullptr.
    void assert_no_locks_locked_except(std::initializer_list<std::variant<void *, lock_enumerator>> canBeLocked);

    bool lock_precedes_lock(const void *old_lock, const void *new_lock);
#else
    static inline void assert_locked(objc_lock_base_t *) {}
    static inline void assert_unlocked(objc_lock_base_t *) {}

    static inline void assert_locked(objc_recursive_lock_base_t *) {}
    static inline void assert_unlocked(objc_recursive_lock_base_t *) {}

    static inline void assert_all_locks_locked() {}
    static inline void assert_no_locks_locked() {}
    static inline void assert_no_locks_locked_except(std::initializer_list<std::variant<void *, lock_enumerator>>) {}
#endif
}

extern const lockdebug::fork_unsafe_t fork_unsafe;

#endif // _OBJC_LOCKDEBUG_H
