/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 26, 2024.
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

#include <memory>

namespace WTF {

enum HashTableDeletedValueType { HashTableDeletedValue };
enum HashTableEmptyValueType { HashTableEmptyValue };

template <typename T> inline T* getPtr(T* p) { return p; }

template <typename T> struct IsSmartPtr {
    static constexpr bool value = false;
    static constexpr bool isNullable = true;
};

template <typename T, bool isSmartPtr>
struct GetPtrHelperBase;

template <typename T>
struct GetPtrHelperBase<T, false /* isSmartPtr */> {
    using PtrType = T*;
    using UnderlyingType = T;
    static T* getPtr(T& p) { return std::addressof(p); }
};

template <typename T>
struct GetPtrHelperBase<T, true /* isSmartPtr */> {
    using PtrType = typename T::PtrType;
    using UnderlyingType = typename T::ValueType;
    static PtrType getPtr(const T& p) { return p.get(); }
};

template <typename T>
struct GetPtrHelper : GetPtrHelperBase<T, IsSmartPtr<T>::value> {
};

template <typename T>
inline typename GetPtrHelper<T>::PtrType getPtr(T& p)
{
    return GetPtrHelper<T>::getPtr(p);
}

template <typename T>
inline typename GetPtrHelper<T>::PtrType getPtr(const T& p)
{
    return GetPtrHelper<T>::getPtr(p);
}

// Explicit specialization for C++ standard library types.

template <typename T, typename Deleter> struct IsSmartPtr<std::unique_ptr<T, Deleter>> {
    static constexpr bool value = true;
    static constexpr bool isNullable = true;
};

template <typename T, typename Deleter>
struct GetPtrHelper<std::unique_ptr<T, Deleter>> {
    using PtrType = T*;
    using UnderlyingType = T;
    static T* getPtr(const std::unique_ptr<T, Deleter>& p) { return p.get(); }
};

} // namespace WTF
