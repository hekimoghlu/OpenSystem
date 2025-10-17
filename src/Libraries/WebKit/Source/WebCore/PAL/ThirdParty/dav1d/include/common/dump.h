/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 3, 2024.
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
#ifndef DAV1D_COMMON_DUMP_H
#define DAV1D_COMMON_DUMP_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "common/bitdepth.h"

static inline void append_plane_to_file(const pixel *buf, ptrdiff_t stride,
                                        int w, int h, const char *const file)
{
    FILE *const f = fopen(file, "ab");
    while (h--) {
        fwrite(buf, w * sizeof(pixel), 1, f);
        buf += PXSTRIDE(stride);
    }
    fclose(f);
}

static inline void hex_fdump(FILE *out, const pixel *buf, ptrdiff_t stride,
                             int w, int h, const char *what)
{
    fprintf(out, "%s\n", what);
    while (h--) {
        int x;
        for (x = 0; x < w; x++)
            fprintf(out, " " PIX_HEX_FMT, buf[x]);
        buf += PXSTRIDE(stride);
        fprintf(out, "\n");
    }
}

static inline void hex_dump(const pixel *buf, ptrdiff_t stride,
                            int w, int h, const char *what)
{
    hex_fdump(stdout, buf, stride, w, h, what);
}

static inline void coef_dump(const coef *buf, const int w, const int h,
                             const int len, const char *what)
{
    int y;
    printf("%s\n", what);
    for (y = 0; y < h; y++) {
        int x;
        for (x = 0; x < w; x++)
            printf(" %*d", len, buf[x]);
        buf += w;
        printf("\n");
    }
}

static inline void ac_dump(const int16_t *buf, int w, int h, const char *what)
{
    printf("%s\n", what);
    while (h--) {
        for (int x = 0; x < w; x++)
            printf(" %03d", buf[x]);
        buf += w;
        printf("\n");
    }
}

#endif /* DAV1D_COMMON_DUMP_H */
