/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 22, 2022.
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
#include "config.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define XXH_INLINE_ALL
#include "xxhash.h"

#include "output/muxer.h"

typedef struct MuxerPriv {
    XXH3_state_t* state;
    FILE *f;
} xxh3Context;

static int xxh3_open(xxh3Context *const xxh3, const char *const file,
                    const Dav1dPictureParameters *const p,
                    const unsigned fps[2])
{
    xxh3->state = XXH3_createState();
    if (!xxh3->state) return DAV1D_ERR(ENOMEM);
    XXH_errorcode err = XXH3_128bits_reset(xxh3->state);
    if (err != XXH_OK) {
        XXH3_freeState(xxh3->state);
        xxh3->state = NULL;
        return DAV1D_ERR(ENOMEM);
    }

    if (!strcmp(file, "-")) {
        xxh3->f = stdout;
    } else if (!(xxh3->f = fopen(file, "wb"))) {
        XXH3_freeState(xxh3->state);
        xxh3->state = NULL;
        fprintf(stderr, "Failed to open %s: %s\n", file, strerror(errno));
        return -1;
    }

    return 0;
}

static int xxh3_write(xxh3Context *const xxh3, Dav1dPicture *const p) {
    const int hbd = p->p.bpc > 8;
    const int w = p->p.w, h = p->p.h;
    uint8_t *yptr = p->data[0];

    for (int y = 0; y < h; y++) {
        XXH3_128bits_update(xxh3->state, yptr, w << hbd);
        yptr += p->stride[0];
    }

    if (p->p.layout != DAV1D_PIXEL_LAYOUT_I400) {
        const int ss_ver = p->p.layout == DAV1D_PIXEL_LAYOUT_I420;
        const int ss_hor = p->p.layout != DAV1D_PIXEL_LAYOUT_I444;
        const int cw = (w + ss_hor) >> ss_hor;
        const int ch = (h + ss_ver) >> ss_ver;
        for (int pl = 1; pl <= 2; pl++) {
            uint8_t *uvptr = p->data[pl];

            for (int y = 0; y < ch; y++) {
                XXH3_128bits_update(xxh3->state, uvptr, cw << hbd);
                uvptr += p->stride[1];
            }
        }
    }

    dav1d_picture_unref(p);

    return 0;
}

static void xxh3_close(xxh3Context *const xxh3) {
    XXH128_hash_t hash = XXH3_128bits_digest(xxh3->state);
    XXH3_freeState(xxh3->state);
    XXH128_canonical_t c;
    XXH128_canonicalFromHash(&c, hash);

    for (int i = 0; i < 16; i++)
        fprintf(xxh3->f, "%2.2x", c.digest[i]);
    fprintf(xxh3->f, "\n");

    if (xxh3->f != stdout)
        fclose(xxh3->f);
}

static int xxh3_verify(xxh3Context *const xxh3, const char * xxh3_str) {
    XXH128_hash_t hash = XXH3_128bits_digest(xxh3->state);
    XXH3_freeState(xxh3->state);

    if (strlen(xxh3_str) < 32)
        return -1;

    XXH128_canonical_t c;
    char t[3] = { 0 };
    for (int i = 0; i < 16; i++) {
        char *ignore;
        memcpy(t, xxh3_str, 2);
        xxh3_str += 2;
        c.digest[i] = (unsigned char) strtoul(t, &ignore, 16);
    }
    XXH128_hash_t verify = XXH128_hashFromCanonical(&c);

    return !XXH128_isEqual(hash, verify);
}

const Muxer xxh3_muxer = {
    .priv_data_size = sizeof(xxh3Context),
    .name = "xxh3",
    .extension = "xxh3",
    .write_header = xxh3_open,
    .write_picture = xxh3_write,
    .write_trailer = xxh3_close,
    .verify = xxh3_verify,
};
