/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 26, 2021.
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
#pragma once

#include <cstdint>
#include <wtf/OptionSet.h>

// GCGL types match the corresponding GL types as defined in OpenGL ES 2.0
// header file gl2.h from khronos.org.
typedef unsigned GCGLenum;
typedef unsigned char GCGLboolean;
typedef unsigned GCGLbitfield;
typedef signed char GCGLbyte;
typedef unsigned char GCGLubyte;
typedef short GCGLshort;
typedef unsigned short GCGLushort;
typedef int GCGLint;
typedef int GCGLsizei;
typedef unsigned GCGLuint;
typedef float GCGLfloat;
typedef unsigned short GCGLhalffloat;
typedef float GCGLclampf;
typedef char GCGLchar;
typedef void* GCGLsync;
typedef void GCGLvoid;

// These GCGL types do not strictly match the GL types as defined in OpenGL ES 2.0
// header file for all platforms.
typedef intptr_t GCGLintptr;
typedef intptr_t GCGLsizeiptr;
typedef intptr_t GCGLvoidptr;
typedef int64_t GCGLint64;
typedef uint64_t GCGLuint64;

typedef GCGLuint PlatformGLObject;

// GCGL types match the corresponding EGL types as defined in Khronos Native
// Platform Graphics Interface - EGL Version 1.5 header file egl.h from
// khronos.org.

// FIXME: These should be renamed to GCEGLxxx
using GCGLDisplay = void*;
using GCGLConfig = void*;
using GCGLContext = void*;
using GCEGLSurface = void*;
using GCGLExternalImage = unsigned;
using GCGLExternalSync = unsigned;

#if !PLATFORM(COCOA)
typedef unsigned GLuint;
#endif

#if ENABLE(WEBXR)
// GL_ANGLE_variable_rasterization_rate_metal
using GCGLMTLRasterizationRateMapANGLE = void*;
#endif

// Order in inverse of in GL specification, so that iteration is in GL specification order.
enum class GCGLErrorCode : uint8_t {
    ContextLost = 1,
    InvalidFramebufferOperation = 1 << 2,
    OutOfMemory = 1 << 3,
    InvalidOperation = 1 << 4,
    InvalidValue = 1 << 5,
    InvalidEnum = 1 << 6
};
using GCGLErrorCodeSet = OptionSet<GCGLErrorCode>;
