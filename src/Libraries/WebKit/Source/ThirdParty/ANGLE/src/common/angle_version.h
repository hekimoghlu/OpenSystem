/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 7, 2023.
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
// Copyright 2014 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// angle_version.h: ANGLE version constants. Generated from git commands.

#ifndef COMMON_ANGLE_VERSION_H_
#define COMMON_ANGLE_VERSION_H_

#include "ANGLEShaderProgramVersion.h"
#include "angle_commit.h"

#define ANGLE_MAJOR_VERSION 2
#define ANGLE_MINOR_VERSION 1

#ifndef ANGLE_REVISION
#    define ANGLE_REVISION ANGLE_COMMIT_POSITION
#endif

#define ANGLE_STRINGIFY(x) #x
#define ANGLE_MACRO_STRINGIFY(x) ANGLE_STRINGIFY(x)

#if (ANGLE_REVISION != 0)
#    define ANGLE_REVISION_SUFFIX "." ANGLE_MACRO_STRINGIFY(ANGLE_REVISION)
#else
#    define ANGLE_REVISION_SUFFIX ""
#endif

#define ANGLE_VERSION_STRING                                             \
    ANGLE_MACRO_STRINGIFY(ANGLE_MAJOR_VERSION)                           \
    "." ANGLE_MACRO_STRINGIFY(ANGLE_MINOR_VERSION) ANGLE_REVISION_SUFFIX \
        " git hash: " ANGLE_COMMIT_HASH

#endif  // COMMON_ANGLE_VERSION_H_
