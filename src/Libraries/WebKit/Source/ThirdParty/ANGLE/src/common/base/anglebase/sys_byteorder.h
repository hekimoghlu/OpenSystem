/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 19, 2025.
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

//
// Copyright 2017 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// sys_byteorder.h: Compatiblity hacks for importing Chromium's base/SHA1.

#ifndef ANGLEBASE_SYS_BYTEORDER_H_
#define ANGLEBASE_SYS_BYTEORDER_H_

#include <cstdlib>

namespace angle
{

namespace base
{

// Returns a value with all bytes in |x| swapped, i.e. reverses the endianness.
inline uint16_t ByteSwap(uint16_t x)
{
#if defined(_MSC_VER)
    return _byteswap_ushort(x);
#else
    return __builtin_bswap16(x);
#endif
}

inline uint32_t ByteSwap(uint32_t x)
{
#if defined(_MSC_VER)
    return _byteswap_ulong(x);
#else
    return __builtin_bswap32(x);
#endif
}

inline uint64_t ByteSwap(uint64_t x)
{
#if defined(_MSC_VER)
    return _byteswap_uint64(x);
#else
    return __builtin_bswap64(x);
#endif
}

}  // namespace base

}  // namespace angle

#endif  // ANGLEBASE_SYS_BYTEORDER_H_
