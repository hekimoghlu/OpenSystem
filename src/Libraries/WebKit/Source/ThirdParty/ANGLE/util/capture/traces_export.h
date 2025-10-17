/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 18, 2024.
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
// restricted_traces_export: Export definitions for restricted traces.

#ifndef ANGLE_TRACES_EXPORT_H_
#define ANGLE_TRACES_EXPORT_H_

// for KHRONOS_APIENTRY
#include <KHR/khrplatform.h>

// See util/util_export.h for details on import/export labels.
#if !defined(ANGLE_TRACE_EXPORT)
#    if defined(_WIN32)
#        if defined(ANGLE_TRACE_IMPLEMENTATION)
#            define ANGLE_TRACE_EXPORT __declspec(dllexport)
#        else
#            define ANGLE_TRACE_EXPORT __declspec(dllimport)
#        endif
#    elif defined(__GNUC__)
#        define ANGLE_TRACE_EXPORT __attribute__((visibility("default")))
#    else
#        define ANGLE_TRACE_EXPORT
#    endif
#endif  // !defined(ANGLE_TRACE_EXPORT)

#if !defined(ANGLE_TRACE_LOADER_EXPORT)
#    if defined(_WIN32)
#        if defined(ANGLE_TRACE_LOADER_IMPLEMENTATION)
#            define ANGLE_TRACE_LOADER_EXPORT __declspec(dllexport)
#        else
#            define ANGLE_TRACE_LOADER_EXPORT __declspec(dllimport)
#        endif
#    elif defined(__GNUC__)
#        define ANGLE_TRACE_LOADER_EXPORT __attribute__((visibility("default")))
#    else
#        define ANGLE_TRACE_LOADER_EXPORT
#    endif
#endif  // !defined(ANGLE_TRACE_LOADER_EXPORT)

#if !defined(ANGLE_REPLAY_EXPORT)
#    if defined(_WIN32)
#        if defined(ANGLE_REPLAY_IMPLEMENTATION)
#            define ANGLE_REPLAY_EXPORT __declspec(dllexport)
#        else
#            define ANGLE_REPLAY_EXPORT __declspec(dllimport)
#        endif
#    elif defined(__GNUC__)
#        define ANGLE_REPLAY_EXPORT __attribute__((visibility("default")))
#    else
#        define ANGLE_REPLAY_EXPORT
#    endif
#endif  // !defined(ANGLE_REPLAY_EXPORT)

typedef void (*GenericProc)(void);
typedef GenericProc(KHRONOS_APIENTRY *LoadProc)(const char *);

#if defined(__cplusplus)
extern "C" {
#endif  // defined(__cplusplus)
ANGLE_TRACE_LOADER_EXPORT void LoadTraceEGL(LoadProc loadProc);
ANGLE_TRACE_LOADER_EXPORT void LoadTraceGLES(LoadProc loadProc);
#if defined(__cplusplus)
}  // extern "C"
#endif  // defined(__cplusplus)

#endif  // ANGLE_TRACES_EXPORT_H_
