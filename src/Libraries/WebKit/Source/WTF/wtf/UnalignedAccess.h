/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 5, 2023.
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

#include <type_traits>
#include <wtf/Platform.h>
#include <wtf/StdLibExtras.h>

namespace WTF {

template<typename Type>
inline Type unalignedLoad(const void* pointer)
{
    static_assert(std::is_trivially_copyable<Type>::value);
    Type result { };
    memcpySpan(asMutableByteSpan(result), unsafeMakeSpan(static_cast<const uint8_t*>(pointer), sizeof(Type)));
    return result;
}

template<typename Type>
inline void unalignedStore(void* pointer, Type value)
{
    static_assert(std::is_trivially_copyable<Type>::value);
    memcpySpan(unsafeMakeSpan(static_cast<uint8_t*>(pointer), sizeof(Type)), asByteSpan(value));
}

} // namespace WTF
