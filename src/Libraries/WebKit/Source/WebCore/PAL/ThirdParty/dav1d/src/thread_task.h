/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 5, 2023.
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
#ifndef DAV1D_SRC_THREAD_TASK_H
#define DAV1D_SRC_THREAD_TASK_H

#include <limits.h>

#include "src/internal.h"

#define FRAME_ERROR (UINT_MAX - 1)
#define TILE_ERROR (INT_MAX - 1)

// these functions assume the task scheduling lock is already taken
int dav1d_task_create_tile_sbrow(Dav1dFrameContext *f, int pass, int cond_signal);
void dav1d_task_frame_init(Dav1dFrameContext *f);

void dav1d_task_delayed_fg(Dav1dContext *c, Dav1dPicture *out, const Dav1dPicture *in);

void *dav1d_worker_task(void *data);

int dav1d_decode_frame_init(Dav1dFrameContext *f);
int dav1d_decode_frame_init_cdf(Dav1dFrameContext *f);
int dav1d_decode_frame_main(Dav1dFrameContext *f);
void dav1d_decode_frame_exit(Dav1dFrameContext *f, int retval);
int dav1d_decode_frame(Dav1dFrameContext *f);
int dav1d_decode_tile_sbrow(Dav1dTaskContext *t);

#endif /* DAV1D_SRC_THREAD_TASK_H */
