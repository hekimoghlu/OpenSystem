/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 18, 2025.
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

#ifndef LIBANGLE_FEATURES_H_
#define LIBANGLE_FEATURES_H_

#include "common/platform.h"

// Feature defaults

// Direct3D9EX
// The "Debug This Pixel..." feature in PIX often fails when using the
// D3D9Ex interfaces.  In order to get debug pixel to work on a Vista/Win 7
// machine, define "ANGLE_D3D9EX=0" in your project file.
#if !defined(ANGLE_D3D9EX)
#    define ANGLE_D3D9EX 1
#endif

// Vsync
// ENABLED allows Vsync to be configured at runtime
// DISABLED disallows Vsync
#if !defined(ANGLE_VSYNC)
#    define ANGLE_VSYNC 1
#endif

// Append HLSL assembly to shader debug info. Defaults to enabled in Debug and off in Release.
#if !defined(ANGLE_APPEND_ASSEMBLY_TO_SHADER_DEBUG_INFO)
#    if !defined(NDEBUG)
#        define ANGLE_APPEND_ASSEMBLY_TO_SHADER_DEBUG_INFO 1
#    else
#        define ANGLE_APPEND_ASSEMBLY_TO_SHADER_DEBUG_INFO 0
#    endif  // !defined(NDEBUG)
#endif      // !defined(ANGLE_APPEND_ASSEMBLY_TO_SHADER_DEBUG_INFO)

// Program link validation of precisions for uniforms. This feature was
// requested by developers to allow non-conformant shaders to be used which
// contain mismatched precisions.
// ENABLED validate that precision for uniforms match between vertex and fragment shaders
// DISABLED allow precision for uniforms to differ between vertex and fragment shaders
#if !defined(ANGLE_PROGRAM_LINK_VALIDATE_UNIFORM_PRECISION)
#    define ANGLE_PROGRAM_LINK_VALIDATE_UNIFORM_PRECISION 1
#endif

// Lose context on Metal command queue error
// ENABLED check Metal command buffer status on completion for error and lose context on error.
// DISABLED Metal backed contexts are never lost.
#if !defined(ANGLE_METAL_LOSE_CONTEXT_ON_ERROR)
#    define ANGLE_METAL_LOSE_CONTEXT_ON_ERROR ANGLE_ENABLED
#endif

#endif  // LIBANGLE_FEATURES_H_
