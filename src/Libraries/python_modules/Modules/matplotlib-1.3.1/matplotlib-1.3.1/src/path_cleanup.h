/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 7, 2024.
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
#ifndef PATH_CLEANUP_H
#define PATH_CLEANUP_H

#include <Python.h>

enum e_snap_mode
{
    SNAP_AUTO,
    SNAP_FALSE,
    SNAP_TRUE
};

void*
get_path_iterator(
    PyObject* path, PyObject* trans, int remove_nans, int do_clip,
    double rect[4], enum e_snap_mode snap_mode, double stroke_width,
    int do_simplify);

unsigned
get_vertex(void* pipeline, double* x, double* y);

void
free_path_iterator(void* pipeline);

#endif /* PATH_CLEANUP_H */
