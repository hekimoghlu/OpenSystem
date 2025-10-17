/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 13, 2023.
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

#include <wtf/CheckedArithmetic.h>
#include <wtf/FastMalloc.h>
#include <wtf/MallocSpan.h>
#include <wtf/Vector.h>

namespace WTF {

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(UniqueArray);
DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(UniqueArrayElement);

template<bool isTriviallyDestructible, typename T> struct UniqueArrayMaker;

template<typename T>
struct UniqueArrayFree {
    static_assert(std::is_trivially_destructible<T>::value);

    void operator()(T* pointer) const
    {
        UniqueArrayMalloc::free(const_cast<typename std::remove_cv<T>::type*>(pointer));
    }
};

template<typename T>
struct UniqueArrayFree<T[]> {
    static_assert(std::is_trivially_destructible<T>::value);

    void operator()(T* pointer) const
    {
        UniqueArrayMalloc::free(const_cast<typename std::remove_cv<T>::type*>(pointer));
    }
};


template<typename T>
struct UniqueArrayMaker<true, T> {
    using ResultType = typename std::unique_ptr<T[], UniqueArrayFree<T[]>>;

    static ResultType make(size_t size)
    {
        // C++ `new T[N]` stores its `N` to somewhere. Otherwise, `delete []` cannot destroy
        // these N elements. But we do not want to increase the size of allocated memory.
        // If it is acceptable, we can just use Vector<T> instead. So this UniqueArray<T> only
        // accepts the type T which has a trivial destructor. This allows us to skip calling
        // destructors for N elements. And this allows UniqueArray<T> not to store its N size.
        static_assert(std::is_trivially_destructible<T>::value);

        // Do not use placement new like `new (storage) T[size]()`. `new T[size]()` requires
        // larger storage than the `sizeof(T) * size` storage since it want to store `size`
        // to somewhere.
        auto storage = MallocSpan<T, UniqueArrayMalloc>::malloc(Checked<size_t>(sizeof(T)) * size);
        VectorTypeOperations<T>::initialize(storage.mutableSpan().data(), storage.mutableSpan().subspan(size).data());
        return ResultType(storage.leakSpan().data());
    }
};

template<typename T>
struct UniqueArrayMaker<false, T> {
    // Since we do not know how to store/retrieve N size to/from allocated memory when calling new [] and delete [],
    // we use new [] and delete [] operators simply. We create UniqueArrayElement container for the type T.
    // UniqueArrayElement has new [] and delete [] operators for FastMalloc. We allocate UniqueArrayElement[] and cast
    // it to T[]. When deleting, the custom deleter casts T[] to UniqueArrayElement[] and deletes it.
    class UniqueArrayElement {
        WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(UniqueArrayElement);
    public:
        struct Deleter {
            void operator()(T* pointer)
            {
                delete [] std::bit_cast<UniqueArrayElement*>(pointer);
            };
        };

        UniqueArrayElement() = default;

        T value { };
    };
    static_assert(sizeof(T) == sizeof(UniqueArrayElement));

    using ResultType = typename std::unique_ptr<T[], typename UniqueArrayElement::Deleter>;

    static ResultType make(size_t size)
    {
        return ResultType(std::bit_cast<T*>(new UniqueArrayElement[size]()));
    }
};

template<typename T>
using UniqueArray = typename UniqueArrayMaker<std::is_trivially_destructible<T>::value, T>::ResultType;

template<typename T>
UniqueArray<T> makeUniqueArray(size_t size)
{
    static_assert(std::is_same<typename std::remove_extent<T>::type, T>::value);
    return UniqueArrayMaker<std::is_trivially_destructible<T>::value, T>::make(size);
}

} // namespace WTF

using WTF::UniqueArray;
using WTF::makeUniqueArray;
