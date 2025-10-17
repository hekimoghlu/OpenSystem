/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 12, 2023.
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

#include "fuchsia_egl.h"
#include "fuchsia_egl_backend.h"

#include <zircon/assert.h>
#include <zircon/syscalls.h>

#include <stdlib.h>

#define FUCHSIA_EGL_WINDOW_MAGIC 0x80738870  // "FXIP"

struct fuchsia_egl_window
{
    uint32_t magic;
    zx_handle_t image_pipe_handle;
    int32_t width;
    int32_t height;
};

fuchsia_egl_window *fuchsia_egl_window_create(zx_handle_t image_pipe_handle,
                                              int32_t width,
                                              int32_t height)
{
    if (width <= 0 || height <= 0)
        return NULL;

    fuchsia_egl_window *egl_window = malloc(sizeof(*egl_window));
    egl_window->magic              = FUCHSIA_EGL_WINDOW_MAGIC;
    egl_window->image_pipe_handle  = image_pipe_handle;
    egl_window->width              = width;
    egl_window->height             = height;
    return egl_window;
}

void fuchsia_egl_window_destroy(fuchsia_egl_window *egl_window)
{
    ZX_ASSERT(egl_window->magic == FUCHSIA_EGL_WINDOW_MAGIC);
    if (egl_window->image_pipe_handle != ZX_HANDLE_INVALID)
    {
        zx_handle_close(egl_window->image_pipe_handle);
        egl_window->image_pipe_handle = ZX_HANDLE_INVALID;
    }
    egl_window->magic = -1U;
    free(egl_window);
}

void fuchsia_egl_window_resize(fuchsia_egl_window *egl_window, int32_t width, int32_t height)
{
    ZX_ASSERT(egl_window->magic == FUCHSIA_EGL_WINDOW_MAGIC);
    if (width <= 0 || height <= 0)
        return;
    egl_window->width  = width;
    egl_window->height = height;
}

int32_t fuchsia_egl_window_get_width(fuchsia_egl_window *egl_window)
{
    ZX_ASSERT(egl_window->magic == FUCHSIA_EGL_WINDOW_MAGIC);
    return egl_window->width;
}

int32_t fuchsia_egl_window_get_height(fuchsia_egl_window *egl_window)
{
    ZX_ASSERT(egl_window->magic == FUCHSIA_EGL_WINDOW_MAGIC);
    return egl_window->height;
}

zx_handle_t fuchsia_egl_window_release_image_pipe(fuchsia_egl_window *egl_window)
{
    ZX_ASSERT(egl_window->magic == FUCHSIA_EGL_WINDOW_MAGIC);
    zx_handle_t image_pipe_handle = egl_window->image_pipe_handle;
    egl_window->image_pipe_handle = ZX_HANDLE_INVALID;
    return image_pipe_handle;
}
