/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 17, 2022.
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

#if CPU(ARM64E)
#include <ptrauth.h>
#endif

#include <type_traits>

namespace WTF {

#if COMPILER_HAS_CLANG_BUILTIN(__builtin_get_vtable_pointer)

template<typename T, typename = std::enable_if_t<std::is_polymorphic_v<T>>>
ALWAYS_INLINE const void* getVTablePointer(const T* o) { return __builtin_get_vtable_pointer(o); }

#else // not COMPILER_HAS_CLANG_BUILTIN(__builtin_get_vtable_pointer)

#if CPU(ARM64E)
template<typename T, typename = std::enable_if_t<std::is_polymorphic_v<T>>>
ALWAYS_INLINE const void* getVTablePointer(const T* o) { return __builtin_ptrauth_auth(*(reinterpret_cast<const void* const*>(o)), ptrauth_key_cxx_vtable_pointer, 0); }
#else // not CPU(ARM64E)
template<typename T, typename = std::enable_if_t<std::is_polymorphic_v<T>>>
ALWAYS_INLINE const void* getVTablePointer(const T* o) { return (*(reinterpret_cast<const void* const*>(o))); }
#endif // not CPU(ARM64E)

#endif // not COMPILER_HAS_CLANG_BUILTIN(__builtin_get_vtable_pointer)

} // namespace WTF

using WTF::getVTablePointer;
