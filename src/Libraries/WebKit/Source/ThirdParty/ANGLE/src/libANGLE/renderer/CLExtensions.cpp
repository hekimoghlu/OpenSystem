/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 14, 2024.
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
// CLExtensions.cpp: Implements the struct methods for CLExtension.

#include "libANGLE/renderer/CLExtensions.h"
#include "libANGLE/renderer/cl_types.h"

#include "common/string_utils.h"

namespace rx
{

CLExtensions::CLExtensions() = default;

CLExtensions::~CLExtensions() = default;

CLExtensions::CLExtensions(CLExtensions &&) = default;

CLExtensions &CLExtensions::operator=(CLExtensions &&) = default;

void CLExtensions::initializeExtensions(std::string &&extensionStr)
{
    ASSERT(extensions.empty());

    extensions.assign(std::move(extensionStr));
    if (extensions.empty())
    {
        return;
    }

    auto hasExtension = [&](const std::string &extension) {
        return angle::ContainsToken(extensions, ' ', extension);
    };

    khrByteAddressableStore       = hasExtension("cl_khr_byte_addressable_store");
    khrGlobalInt32BaseAtomics     = hasExtension("cl_khr_global_int32_base_atomics");
    khrGlobalInt32ExtendedAtomics = hasExtension("cl_khr_global_int32_extended_atomics");
    khrLocalInt32BaseAtomics      = hasExtension("cl_khr_local_int32_base_atomics");
    khrLocalInt32ExtendedAtomics  = hasExtension("cl_khr_local_int32_extended_atomics");

    khr3D_ImageWrites     = hasExtension("cl_khr_3d_image_writes");
    khrDepthImages        = hasExtension("cl_khr_depth_images");
    khrImage2D_FromBuffer = hasExtension("cl_khr_image2d_from_buffer");

    khrExtendedVersioning   = hasExtension("cl_khr_extended_versioning");
    khrFP64                 = hasExtension("cl_khr_fp64");
    khrICD                  = hasExtension("cl_khr_icd");
    khrInt64BaseAtomics     = hasExtension("cl_khr_int64_base_atomics");
    khrInt64ExtendedAtomics = hasExtension("cl_khr_int64_extended_atomics");
}

void CLExtensions::initializeVersionedExtensions(const NameVersionVector &versionedExtList)
{
    ASSERT(extensionsWithVersion.empty());
    ASSERT(extensions.empty());

    extensionsWithVersion = std::move(versionedExtList);

    std::string extensionString = "";
    for (cl_name_version ext : extensionsWithVersion)
    {
        extensionString += ext.name;
        extensionString += " ";
    }

    return initializeExtensions(std::move(extensionString));
}

}  // namespace rx
