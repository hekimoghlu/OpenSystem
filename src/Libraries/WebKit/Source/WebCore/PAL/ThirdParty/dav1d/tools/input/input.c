/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 19, 2023.
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

#include "common/attributes.h"
#include "common/intops.h"

#include "input/input.h"
#include "input/demuxer.h"

struct DemuxerContext {
    DemuxerPriv *data;
    const Demuxer *impl;
};

extern const Demuxer ivf_demuxer;
extern const Demuxer annexb_demuxer;
extern const Demuxer section5_demuxer;
static const Demuxer *const demuxers[] = {
    &ivf_demuxer,
    &annexb_demuxer,
    &section5_demuxer,
    NULL
};

int input_open(DemuxerContext **const c_out,
               const char *const name, const char *const filename,
               unsigned fps[2], unsigned *const num_frames, unsigned timebase[2])
{
    const Demuxer *impl;
    DemuxerContext *c;
    int res, i;

    if (name) {
        for (i = 0; demuxers[i]; i++) {
            if (!strcmp(demuxers[i]->name, name)) {
                impl = demuxers[i];
                break;
            }
        }
        if (!demuxers[i]) {
            fprintf(stderr, "Failed to find demuxer named \"%s\"\n", name);
            return DAV1D_ERR(ENOPROTOOPT);
        }
    } else {
        int probe_sz = 0;
        for (i = 0; demuxers[i]; i++)
            probe_sz = imax(probe_sz, demuxers[i]->probe_sz);
        uint8_t *const probe_data = malloc(probe_sz);
        if (!probe_data) {
            fprintf(stderr, "Failed to allocate memory\n");
            return DAV1D_ERR(ENOMEM);
        }
        FILE *f = fopen(filename, "rb");
        if (!f) {
            fprintf(stderr, "Failed to open input file %s: %s\n", filename, strerror(errno));
            return errno ? DAV1D_ERR(errno) : DAV1D_ERR(EIO);
        }
        res = !!fread(probe_data, 1, probe_sz, f);
        fclose(f);
        if (!res) {
            free(probe_data);
            fprintf(stderr, "Failed to read probe data\n");
            return errno ? DAV1D_ERR(errno) : DAV1D_ERR(EIO);
        }

        for (i = 0; demuxers[i]; i++) {
            if (demuxers[i]->probe(probe_data)) {
                impl = demuxers[i];
                break;
            }
        }
        free(probe_data);
        if (!demuxers[i]) {
            fprintf(stderr,
                    "Failed to probe demuxer for file %s\n",
                    filename);
            return DAV1D_ERR(ENOPROTOOPT);
        }
    }

    if (!(c = calloc(1, sizeof(DemuxerContext) + impl->priv_data_size))) {
        fprintf(stderr, "Failed to allocate memory\n");
        return DAV1D_ERR(ENOMEM);
    }
    c->impl = impl;
    c->data = (DemuxerPriv *) &c[1];
    if ((res = impl->open(c->data, filename, fps, num_frames, timebase)) < 0) {
        free(c);
        return res;
    }
    *c_out = c;

    return 0;
}

int input_read(DemuxerContext *const ctx, Dav1dData *const data) {
    return ctx->impl->read(ctx->data, data);
}

int input_seek(DemuxerContext *const ctx, const uint64_t pts) {
    return ctx->impl->seek ? ctx->impl->seek(ctx->data, pts) : -1;
}

void input_close(DemuxerContext *const ctx) {
    ctx->impl->close(ctx->data);
    free(ctx);
}
