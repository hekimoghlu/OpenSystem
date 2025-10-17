/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 20, 2024.
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

// Copyright 2019 The Fuchsia Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef FUCHSIA_EGL_BACKEND_H_
#define FUCHSIA_EGL_BACKEND_H_

#include <zircon/types.h>

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(FUCHSIA_EGL_EXPORT)
#    define FUCHSIA_EGL_EXPORT __attribute__((__visibility__("default")))
#endif

FUCHSIA_EGL_EXPORT
zx_handle_t fuchsia_egl_window_release_image_pipe(fuchsia_egl_window *egl_window);

#ifdef __cplusplus
}
#endif

#endif  // FUCHSIA_EGL_BACKEND_H_
