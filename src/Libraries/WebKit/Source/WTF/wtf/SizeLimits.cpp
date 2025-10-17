/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 8, 2024.
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
#include "config.h"

#include <type_traits>
#include <utility>
#include <wtf/Assertions.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/Vector.h>

namespace WTF {

#if ASSERT_ENABLED || ENABLE(SECURITY_ASSERTIONS)
struct SameSizeAsRefCounted {
    int a;
    bool b;
    bool c;
    bool d;
    bool e;
    // The debug version may get bigger.
};
#else
struct SameSizeAsRefCounted {
    int a;
    // Don't add anything here because this should stay small.
};
#endif

static_assert(sizeof(RefCounted<int>) == sizeof(SameSizeAsRefCounted), "RefCounted should stay small!");
static_assert(sizeof(RefPtr<RefCounted<int>>) == sizeof(int*), "RefPtr should stay small!");

#if !ASAN_ENABLED
template<typename T, unsigned inlineCapacity = 0>
struct SameSizeAsVectorWithInlineCapacity;

template<typename T>
struct SameSizeAsVectorWithInlineCapacity<T, 0> {
    WTF_MAKE_NONCOPYABLE(SameSizeAsVectorWithInlineCapacity);
public:
    void* bufferPointer;
    unsigned capacity;
    unsigned size;
};

template<typename T>
struct SameSizeAsVectorWithInlineCapacityBase : SameSizeAsVectorWithInlineCapacity<T> {
};

template<typename T, unsigned inlineCapacity>
struct SameSizeAsVectorWithInlineCapacity : SameSizeAsVectorWithInlineCapacityBase<T> {
    WTF_MAKE_NONCOPYABLE(SameSizeAsVectorWithInlineCapacity);
public:
    ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    typename std::aligned_storage<sizeof(T), std::alignment_of<T>::value>::type inlineBuffer[inlineCapacity];
    ALLOW_DEPRECATED_DECLARATIONS_END
};

static_assert(sizeof(Vector<int>) == sizeof(SameSizeAsVectorWithInlineCapacity<int>), "Vector should stay small!");
static_assert(sizeof(Vector<int, 1>) == sizeof(SameSizeAsVectorWithInlineCapacity<int, 1>), "Vector should stay small!");
static_assert(sizeof(Vector<int, 2>) == sizeof(SameSizeAsVectorWithInlineCapacity<int, 2>), "Vector should stay small!");
static_assert(sizeof(Vector<int, 3>) == sizeof(SameSizeAsVectorWithInlineCapacity<int, 3>), "Vector should stay small!");
#endif
}
