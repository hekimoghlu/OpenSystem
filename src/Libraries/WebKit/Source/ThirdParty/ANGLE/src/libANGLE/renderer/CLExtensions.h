/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 28, 2024.
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
// CLExtensions.h: Defines the rx::CLExtensions struct.

#ifndef LIBANGLE_RENDERER_CLEXTENSIONS_H_
#define LIBANGLE_RENDERER_CLEXTENSIONS_H_

#include "libANGLE/renderer/cl_types.h"

namespace rx
{

struct CLExtensions
{
    CLExtensions();
    ~CLExtensions();

    CLExtensions(const CLExtensions &)            = delete;
    CLExtensions &operator=(const CLExtensions &) = delete;

    CLExtensions(CLExtensions &&);
    CLExtensions &operator=(CLExtensions &&);

    void initializeExtensions(std::string &&extensionStr);
    void initializeVersionedExtensions(const NameVersionVector &versionedExtList);

    std::string versionStr;
    cl_version version = 0u;

    std::string extensions;
    NameVersionVector extensionsWithVersion;

    // These Khronos extension names must be returned by all devices that support OpenCL 1.1.
    bool khrByteAddressableStore       = false;  // cl_khr_byte_addressable_store
    bool khrGlobalInt32BaseAtomics     = false;  // cl_khr_global_int32_base_atomics
    bool khrGlobalInt32ExtendedAtomics = false;  // cl_khr_global_int32_extended_atomics
    bool khrLocalInt32BaseAtomics      = false;  // cl_khr_local_int32_base_atomics
    bool khrLocalInt32ExtendedAtomics  = false;  // cl_khr_local_int32_extended_atomics

    // These Khronos extension names must be returned by all devices that support
    // OpenCL 2.0, OpenCL 2.1, or OpenCL 2.2. For devices that support OpenCL 3.0, these
    // extension names must be returned when and only when the optional feature is supported.
    bool khr3D_ImageWrites     = false;  // cl_khr_3d_image_writes
    bool khrDepthImages        = false;  // cl_khr_depth_images
    bool khrImage2D_FromBuffer = false;  // cl_khr_image2d_from_buffer

    // Optional extensions
    bool khrExtendedVersioning   = false;  // cl_khr_extended_versioning
    bool khrFP64                 = false;  // cl_khr_fp64
    bool khrICD                  = false;  // cl_khr_icd
    bool khrInt64BaseAtomics     = false;  // cl_khr_int64_base_atomics
    bool khrInt64ExtendedAtomics = false;  // cl_khr_int64_extended_atomics
};

}  // namespace rx

#endif  // LIBANGLE_RENDERER_CLEXTENSIONS_H_
