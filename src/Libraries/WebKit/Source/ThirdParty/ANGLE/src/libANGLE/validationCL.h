/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 1, 2022.
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

// validationCL.h: Validation functions for generic CL entry point parameters

#ifndef LIBANGLE_VALIDATIONCL_H_
#define LIBANGLE_VALIDATIONCL_H_

#include "libANGLE/CLBuffer.h"
#include "libANGLE/CLCommandQueue.h"
#include "libANGLE/CLContext.h"
#include "libANGLE/CLDevice.h"
#include "libANGLE/CLEvent.h"
#include "libANGLE/CLImage.h"
#include "libANGLE/CLKernel.h"
#include "libANGLE/CLMemory.h"
#include "libANGLE/CLPlatform.h"
#include "libANGLE/CLProgram.h"
#include "libANGLE/CLSampler.h"

#define ANGLE_CL_VALIDATE_VOID(EP, ...)              \
    do                                               \
    {                                                \
        if (Validate##EP(__VA_ARGS__) != CL_SUCCESS) \
        {                                            \
            return;                                  \
        }                                            \
    } while (0)

#define ANGLE_CL_VALIDATE_ERROR(EP, ...)              \
    do                                                \
    {                                                 \
        cl_int errorCode = Validate##EP(__VA_ARGS__); \
        if (errorCode != CL_SUCCESS)                  \
        {                                             \
            return errorCode;                         \
        }                                             \
    } while (0)

#define ANGLE_CL_VALIDATE_ERRCODE_RET(EP, ...)        \
    do                                                \
    {                                                 \
        cl_int errorCode = Validate##EP(__VA_ARGS__); \
        if (errorCode != CL_SUCCESS)                  \
        {                                             \
            if (errcode_ret != nullptr)               \
            {                                         \
                *errcode_ret = errorCode;             \
            }                                         \
            return nullptr;                           \
        }                                             \
    } while (0)

#define ANGLE_CL_VALIDATE_POINTER(EP, ...)            \
    do                                                \
    {                                                 \
        cl_int errorCode = Validate##EP(__VA_ARGS__); \
        if (errorCode != CL_SUCCESS)                  \
        {                                             \
            return nullptr;                           \
        }                                             \
    } while (0)

#endif  // LIBANGLE_VALIDATIONCL_H_
