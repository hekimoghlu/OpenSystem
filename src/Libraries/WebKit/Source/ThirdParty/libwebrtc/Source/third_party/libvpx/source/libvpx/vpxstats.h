/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 18, 2022.
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
#ifndef VPX_VPXSTATS_H_
#define VPX_VPXSTATS_H_

#include <stdio.h>

#include "vpx/vpx_encoder.h"

#ifdef __cplusplus
extern "C" {
#endif

/* This structure is used to abstract the different ways of handling
 * first pass statistics
 */
typedef struct {
  vpx_fixed_buf_t buf;
  int pass;
  FILE *file;
  char *buf_ptr;
  size_t buf_alloc_sz;
} stats_io_t;

int stats_open_file(stats_io_t *stats, const char *fpf, int pass);
int stats_open_mem(stats_io_t *stats, int pass);
void stats_close(stats_io_t *stats, int last_pass);
void stats_write(stats_io_t *stats, const void *pkt, size_t len);
vpx_fixed_buf_t stats_get(stats_io_t *stats);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VPX_VPXSTATS_H_
