/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 19, 2024.
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
// dispatch.cpp: Implements a function to fetch the ANGLE OpenCL dispatch table.

#include "libOpenCL/dispatch.h"

#include "anglebase/no_destructor.h"
#include "common/debug.h"
#include "common/system_utils.h"

#include <memory>

#ifdef _WIN32
#    include <windows.h>
#endif

namespace cl
{

namespace
{

std::unique_ptr<angle::Library> &EntryPointsLib()
{
    static angle::base::NoDestructor<std::unique_ptr<angle::Library>> sEntryPointsLib;
    return *sEntryPointsLib;
}

IcdDispatch CreateDispatch()
{
    const cl_icd_dispatch *clIcdDispatch = nullptr;
    const char *error                    = nullptr;

    // Try to find ANGLE's GLESv2 library in the consistent way, which might fail
    // if the current library or a link to it is not in ANGLE's binary directory
    EntryPointsLib().reset(
        angle::OpenSharedLibrary(ANGLE_GLESV2_LIBRARY_NAME, angle::SearchType::ModuleDir));
    if (EntryPointsLib() && EntryPointsLib()->getNative() != nullptr)
    {
        EntryPointsLib()->getAs("gCLIcdDispatchTable", &clIcdDispatch);
        if (clIcdDispatch == nullptr)
        {
            INFO() << "Found system's instead of ANGLE's GLESv2 library";
        }
    }
    else
    {
        error = "Not able to find GLESv2 library";
    }

    // If not found try to find ANGLE's GLESv2 library in build path
    if (clIcdDispatch == nullptr)
    {
#ifdef _WIN32
        // On Windows the build path 'ANGLE_GLESV2_LIBRARY_PATH' is provided by the build system
        const char path[] = ANGLE_GLESV2_LIBRARY_PATH "\\" ANGLE_GLESV2_LIBRARY_NAME ".dll";
        // This function allows to load further dependent libraries from the same directory
        HMODULE handle = LoadLibraryExA(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (handle != nullptr)
        {
            clIcdDispatch = reinterpret_cast<const cl_icd_dispatch *>(
                GetProcAddress(handle, "gCLIcdDispatchTable"));
            if (clIcdDispatch == nullptr)
            {
                error = "Error loading CL dispatch table.";
            }
        }
#else
        // On posix-compatible systems this will also search in the rpath, which is the build path
        EntryPointsLib().reset(
            angle::OpenSharedLibrary(ANGLE_GLESV2_LIBRARY_NAME, angle::SearchType::SystemDir));
        if (EntryPointsLib() && EntryPointsLib()->getNative() != nullptr)
        {
            EntryPointsLib()->getAs("gCLIcdDispatchTable", &clIcdDispatch);
            if (clIcdDispatch == nullptr)
            {
                INFO() << "Found system's instead of ANGLE's GLESv2 library";
            }
        }
#endif
    }

    IcdDispatch dispatch;
    if (clIcdDispatch != nullptr)
    {
        static_cast<cl_icd_dispatch &>(dispatch) = *clIcdDispatch;
        dispatch.clIcdGetPlatformIDsKHR          = reinterpret_cast<clIcdGetPlatformIDsKHR_fn>(
            clIcdDispatch->clGetExtensionFunctionAddress("clIcdGetPlatformIDsKHR"));
    }
    else if (error != nullptr)
    {
        ERR() << error;
    }
    return dispatch;
}

}  // anonymous namespace

const IcdDispatch &GetDispatch()
{
    static const IcdDispatch sDispatch(CreateDispatch());
    return sDispatch;
}

}  // namespace cl
