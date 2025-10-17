/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 25, 2023.
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
// cl_util.h: Helper functions for the CL back end

#ifndef LIBANGLE_RENDERER_CL_CL_UTIL_H_
#define LIBANGLE_RENDERER_CL_CL_UTIL_H_

#include "libANGLE/renderer/cl_types.h"

#include "anglebase/no_destructor.h"

#include <unordered_set>

namespace rx
{

// Extract numeric version from OpenCL version string
cl_version ExtractCLVersion(const std::string &version);

using CLExtensionSet = std::unordered_set<std::string>;

// Get a set of OpenCL extensions which are supported to be passed through
inline const CLExtensionSet &GetSupportedCLExtensions()
{
    static angle::base::NoDestructor<const CLExtensionSet> sExtensions({
        // clang-format off

        // These Khronos extension names must be returned by all devices that support OpenCL 1.1.
        "cl_khr_byte_addressable_store",
        "cl_khr_global_int32_base_atomics",
        "cl_khr_global_int32_extended_atomics",
        "cl_khr_local_int32_base_atomics",
        "cl_khr_local_int32_extended_atomics",

        // These Khronos extension names must be returned by all devices that support
        // OpenCL 2.0, OpenCL 2.1, or OpenCL 2.2. For devices that support OpenCL 3.0, these
        // extension names must be returned when and only when the optional feature is supported.
        "cl_khr_3d_image_writes",
        "cl_khr_depth_images",
        "cl_khr_image2d_from_buffer",

        // Optional extensions
        "cl_khr_extended_versioning",
        "cl_khr_fp64",
        "cl_khr_icd",
        "cl_khr_int64_base_atomics",
        "cl_khr_int64_extended_atomics"

        // clang-format on
    });
    return *sExtensions;
}

// Check if a specific OpenCL extensions is supported to be passed through
inline bool IsCLExtensionSupported(const std::string &extension)
{
    const CLExtensionSet &supported = GetSupportedCLExtensions();
    return supported.find(extension) != supported.cend();
}

// Filter out extensions which are not (yet) supported to be passed through
void RemoveUnsupportedCLExtensions(std::string &extensions);
void RemoveUnsupportedCLExtensions(NameVersionVector &extensions);

}  // namespace rx

#endif  // LIBANGLE_RENDERER_CL_CL_UTIL_H_
