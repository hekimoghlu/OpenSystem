/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 20, 2024.
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

#include "BInline.h"
#include "Mutex.h"
#include "Sizes.h"

#if !BUSE(LIBPAS)

namespace bmalloc {

// Usage:
//     Object* object = PerProcess<Object>::get();
//     x = object->field->field;
//
// Object will be instantiated only once, even in the face of concurrency.
//
// NOTE: If you observe global side-effects of the Object constructor, be
// sure to lock the Object mutex. For example:
//
// Object() : m_field(...) { globalFlag = true }
//
// Object* object = PerProcess<Object>::get();
// x = object->m_field; // OK
// if (globalFlag) { ... } // Undefined behavior.
//
// LockHolder lock(PerProcess<Object>::mutex());
// Object* object = PerProcess<Object>::get(lock);
// if (globalFlag) { ... } // OK.

struct PerProcessData {
    const char* disambiguator;
    void* memory;
    size_t size;
    size_t alignment;
    Mutex mutex;
    bool isInitialized;
    PerProcessData* next;
};

constexpr unsigned stringHash(const char* string)
{
    unsigned result = 5381;
    while (char c = *string++)
        result = result * 33 + c;
    return result;
}

BEXPORT PerProcessData* getPerProcessData(unsigned disambiguatorHash, const char* disambiguator, size_t size, size_t alignment);

template<typename T>
class PerProcess {
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
        return s_object.load(std::memory_order_relaxed);
    }
    
    static Mutex& mutex()
    {
        if (!s_data)
            coalesce();
        return s_data->mutex;
    }

private:
    static void coalesce()
    {
        if (s_data)
            return;
        
        const char* disambiguator = BFUNCTION_SIGNATURE;
        s_data = getPerProcessData(stringHash(disambiguator), disambiguator, sizeof(T), std::alignment_of<T>::value);
    }
    
    BNO_INLINE static T* getSlowCase()
    {
        LockHolder lock(mutex());
        if (!s_object.load()) {
            if (s_data->isInitialized)
                s_object.store(static_cast<T*>(s_data->memory));
            else {
                T* t = new (s_data->memory) T(lock);
                s_object.store(t);
                s_data->isInitialized = true;
            }
        }
        return s_object.load();
    }

    static std::atomic<T*> s_object;
    static PerProcessData* s_data;
};

template<typename T>
std::atomic<T*> PerProcess<T>::s_object { nullptr };

template<typename T>
PerProcessData* PerProcess<T>::s_data { nullptr };

} // namespace bmalloc

#endif
