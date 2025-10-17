/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 28, 2025.
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

#ifndef AGG_CONFIG_INCLUDED
#define AGG_CONFIG_INCLUDED

// This file can be used to redefine certain data types.

//---------------------------------------
// 1. Default basic types such as:
// 
// AGG_INT8
// AGG_INT8U
// AGG_INT16
// AGG_INT16U
// AGG_INT32
// AGG_INT32U
// AGG_INT64
// AGG_INT64U
//
// Just replace this file with new defines if necessary. 
// For example, if your compiler doesn't have a 64 bit integer type
// you can still use AGG if you define the follows:
//
// #define AGG_INT64  int
// #define AGG_INT64U unsigned
//
// It will result in overflow in 16 bit-per-component image/pattern resampling
// but it won't result any crash and the rest of the library will remain 
// fully functional.


//---------------------------------------
// 2. Default rendering_buffer type. Can be:
//
// Provides faster access for massive pixel operations, 
// such as blur, image filtering:
// #define AGG_RENDERING_BUFFER row_ptr_cache<int8u>
// 
// Provides cheaper creation and destruction (no mem allocs):
// #define AGG_RENDERING_BUFFER row_accessor<int8u>
//
// You can still use both of them simultaneouslyin your applications 
// This #define is used only for default rendering_buffer type,
// in short hand typedefs like pixfmt_rgba32.

#endif
