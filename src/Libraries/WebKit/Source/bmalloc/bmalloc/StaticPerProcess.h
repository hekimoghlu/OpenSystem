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

#include "BExport.h"
#include "BInline.h"
#include "Mutex.h"
#include "Sizes.h"

namespace bmalloc {

// StaticPerProcess<T> behaves like PerProcess<T>, but we need to explicitly define storage for T with EXTERN.
// In this way, we allocate storage for a per-process object statically instead of allocating memory at runtime.
// To enforce this, we have DECLARE and DEFINE macros. If you do not know about T of StaticPerProcess<T>, you should use PerProcess<T> instead.
//
// Usage:
//     In Object.h
//         class Object : public StaticPerProcess<Object> {
//             ...
//         };
//         DECLARE_STATIC_PER_PROCESS_STORAGE(Object);
//
//     In Object.cpp
//         DEFINE_STATIC_PER_PROCESS_STORAGE(Object);
//
//     Object* object = Object::get();
//     x = object->field->field;
//
// Object will be instantiated only once, even in the presence of concurrency.
//
template<typename T> struct StaticPerProcessStorageTraits;

template<typename T>
class StaticPerProcess {
public:
    static T* get()
    {
        T* object = getFastCase();
        if (!object)
            return getSlowCase();
        return object;
    }

    static T* getFastCase()
    {
        using Storage = typename StaticPerProcessStorageTraits<T>::Storage;
        return (Storage::s_object).load(std::memory_order_relaxed);
    }

    static Mutex& mutex()
    {
        using Storage = typename StaticPerProcessStorageTraits<T>::Storage;
        return Storage::s_mutex;
    }

private:
    BNO_INLINE static T* getSlowCase()
    {
        using Storage = typename StaticPerProcessStorageTraits<T>::Storage;
        LockHolder lock(Storage::s_mutex);
        T* t = Storage::s_object.load(std::memory_order_consume);
        if (!t) {
            t = new (&Storage::s_memory) T(lock);
            Storage::s_object.store(t, std::memory_order_release);
        }
        return t;
    }
};

#define DECLARE_STATIC_PER_PROCESS_STORAGE(Type) DECLARE_STATIC_PER_PROCESS_STORAGE_WITH_LINKAGE(Type, BEXPORT)

// Use instead of DECLARE_STATIC_PER_PROCESS_STORAGE when the type being
// instantiated does not have export symbol visibility.
#define DECLARE_STATIC_PER_PROCESS_STORAGE_WITH_LINKAGE(Type, Linkage) \
template<> struct StaticPerProcessStorageTraits<Type> { \
    using Memory = typename std::aligned_storage<sizeof(Type), std::alignment_of<Type>::value>::type; \
    struct Linkage Storage { \
        static std::atomic<Type*> s_object; \
        static Mutex s_mutex; \
        static Memory s_memory; \
    }; \
};

#define DEFINE_STATIC_PER_PROCESS_STORAGE(Type) \
    std::atomic<Type*> StaticPerProcessStorageTraits<Type>::Storage::s_object { nullptr }; \
    Mutex StaticPerProcessStorageTraits<Type>::Storage::s_mutex { }; \
    StaticPerProcessStorageTraits<Type>::Memory StaticPerProcessStorageTraits<Type>::Storage::s_memory { };

} // namespace bmalloc
