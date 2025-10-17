/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 22, 2024.
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
// Copyright 2021 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// entry_points_cl_utils.h: These helpers are used in CL entry point routines.

#ifndef LIBGLESV2_ENTRY_POINTS_CL_UTILS_H_
#define LIBGLESV2_ENTRY_POINTS_CL_UTILS_H_

#include "libANGLE/CLBitField.h"
#include "libANGLE/Debug.h"

#include "common/PackedCLEnums_autogen.h"

#include <cinttypes>
#include <cstdio>

#if defined(ANGLE_ENABLE_DEBUG_TRACE)
#    define CL_EVENT(entryPoint, ...)                    \
        std::printf("CL " #entryPoint ": " __VA_ARGS__); \
        std::printf("\n")
#else
#    define CL_EVENT(entryPoint, ...) (void(0))
#endif

namespace cl
{

// Handling packed enums
template <typename PackedT, typename FromT>
typename std::enable_if_t<std::is_enum<PackedT>::value, PackedT> PackParam(FromT from)
{
    return FromCLenum<PackedT>(from);
}

// Handling bit fields
template <typename PackedT, typename FromT>
typename std::enable_if_t<std::is_same<PackedT, BitField>::value, PackedT> PackParam(FromT from)
{
    return PackedT(from);
}

void InitBackEnds(bool isIcd);

}  // namespace cl

#endif  // LIBGLESV2_ENTRY_POINTS_CL_UTILS_H_
