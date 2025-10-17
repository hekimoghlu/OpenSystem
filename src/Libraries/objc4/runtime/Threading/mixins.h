/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 7, 2025.
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
* mixins.h
* Thread related utility mixins
**********************************************************************/

#ifndef _OBJC_THREAD_MIXINS_H
#define _OBJC_THREAD_MIXINS_H

// .. locker_mixin .....................................................

// Adds locker, conditional_locker, lockWith(), unlockWith(), lockTwo()
// and unlockTwo() to a class.
template <class T>
class locker_mixin: public T {
public:
    using T::T;

    // Address-ordered lock discipline for a pair of locks.
    void lockWith(T& other) {
        if (this < &other) {
            T::lock();
            other.lock();
        } else {
            other.lock();
            if (this != &other) T::lock();
        }
    }

    void unlockWith(T& other) {
        T::unlock();
        if (this != &other) other.unlock();
    }

    static void lockTwo(locker_mixin *lock1, locker_mixin *lock2) {
        lock1->lockWith(*lock2);
    }

    static void unlockTwo(locker_mixin *lock1, locker_mixin *lock2) {
        lock1->unlockWith(*lock2);
    }

    // Scoped lock and unlock
    class locker : nocopy_t {
        T& lock;
    public:
        locker(T& newLock) : lock(newLock) { lock.lock(); }
        ~locker() { lock.unlock(); }
    };

    // Either scoped lock and unlock, or NOP.
    class conditional_locker : nocopy_t {
        T& lock;
        bool didLock;
    public:
        conditional_locker(T& newLock, bool shouldLock)
            : lock(newLock), didLock(shouldLock)
        {
            if (shouldLock) lock.lock();
        }
        ~conditional_locker() { if (didLock) lock.unlock(); }
    };
};

// .. getter_setter ....................................................

// Adds implementations of the essential operators for arithmetic and
// pointer types

template <class T, class Enable=void>
class getter_setter {};

// For arithmetic types
template <class T>
class getter_setter<T,
                    typename std::enable_if<std::is_arithmetic<
                                                typename T::type>::value>::type> : T {
public:
    using type = typename T::type;

    ALWAYS_INLINE getter_setter& operator=(const type &value) {
        T::set_(value);
        return *this;
    }
    ALWAYS_INLINE getter_setter& operator+=(const type &value) {
        T::set_(T::get_() + value);
        return *this;
    }
    ALWAYS_INLINE getter_setter& operator-=(const type &value) {
        T::set_(T::get_() - value);
        return *this;
    }
    ALWAYS_INLINE getter_setter& operator*=(const type &value) {
        T::set_(T::get_() * value);
        return *this;
    }
    ALWAYS_INLINE getter_setter& operator/=(const type &value) {
        T::set_(T::get_() / value);
        return *this;
    }
    ALWAYS_INLINE getter_setter& operator%=(const type &value) {
        T::set_(T::get_() % value);
        return *this;
    }
    ALWAYS_INLINE getter_setter& operator&=(const type &value) {
        T::set_(T::get_() & value);
        return *this;
    }
    ALWAYS_INLINE getter_setter& operator|=(const type &value) {
        T::set_(T::get_() | value);
        return *this;
    }
    ALWAYS_INLINE getter_setter& operator^=(const type &value) {
        T::set_(T::get_() ^ value);
        return *this;
    }
    ALWAYS_INLINE getter_setter& operator<<=(const type &value) {
        T::set_(T::get_() << value);
        return *this;
    }
    ALWAYS_INLINE getter_setter& operator>>=(const type &value) {
        T::set_(T::get_() >> value);
        return *this;
    }

    ALWAYS_INLINE type operator++() {
        type result = T::get_() + 1;
        T::set_(result);
        return result;
    }
    ALWAYS_INLINE type operator++(int) {
        type result = T::get_();
        T::set_(result + 1);
        return result;
    }

    ALWAYS_INLINE type operator--() {
        type result = T::get_() - 1;
        T::set_(result);
        return result;
    }
    ALWAYS_INLINE type operator--(int) {
        type result = T::get_();
        T::set_(result - 1);
        return result;
    }

    ALWAYS_INLINE operator type() const {
        return T::get_();
    }
};

// For non-void pointer types
template <class T>
class getter_setter<T,
                    typename std::enable_if<std::is_pointer<
                                                typename T::type>::value
                                            && !std::is_void<
                                                typename std::remove_pointer<typename T::type>::type>
                                            ::value>::type> : T {
public:
    using type = typename T::type;
    using points_to_type = typename std::remove_pointer<typename T::type>::type;

    ALWAYS_INLINE getter_setter& operator=(const type &value) {
        T::set_(value);
        return *this;
    }
    ALWAYS_INLINE getter_setter& operator+=(std::ptrdiff_t value) {
        T::set_(T::get_() + value);
        return *this;
    }
    ALWAYS_INLINE getter_setter& operator-=(std::ptrdiff_t value) {
        T::set_(T::get_() - value);
        return *this;
    }

    ALWAYS_INLINE type operator++() {
        type result = T::get_() + 1;
        T::set_(result);
        return result;
    }
    ALWAYS_INLINE type operator++(int) {
        type result = T::get_();
        T::set_(result + 1);
        return result;
    }

    ALWAYS_INLINE type operator--() {
        type result = T::get_() - 1;
        T::set_(result);
        return result;
    }
    ALWAYS_INLINE type operator--(int) {
        type result = T::get_();
        T::set_(result - 1);
        return result;
    }

    ALWAYS_INLINE operator type() const {
        return T::get_();
    }

    ALWAYS_INLINE points_to_type& operator[](std::size_t ndx) {
        return T::get_()[ndx];
    }
    ALWAYS_INLINE points_to_type& operator*() {
        return *T::get_();
    }
    ALWAYS_INLINE points_to_type* operator->() {
        return T::get_();
    }
};

// For void *
template <class T>
class getter_setter<T,
                    typename std::enable_if<std::is_pointer<
                                                typename T::type>::value
                                            && std::is_void<
                                                typename std::remove_pointer<typename T::type>::type>
                                            ::value>::type> : T {
public:
    using type = typename T::type;
    using points_to_type = typename std::remove_pointer<typename T::type>::type;

    ALWAYS_INLINE getter_setter& operator=(const type &value) {
        T::set_(value);
        return *this;
    }

    ALWAYS_INLINE operator type() const {
        return T::get_();
    }
};

#endif // _OBJC_THREAD_MIXINS_H
