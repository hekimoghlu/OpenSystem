/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 20, 2023.
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

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include "dav1d/headers.h"

#include "input/demuxer.h"
#include "input/parse.h"

#define PROBE_SIZE 1024

static int section5_probe(const uint8_t *data) {
    int ret, cnt = 0;

    // Check that the first OBU is a Temporal Delimiter.
    size_t obu_size;
    enum Dav1dObuType type;
    ret = parse_obu_header(data + cnt, PROBE_SIZE - cnt,
                           &obu_size, &type, 0);
    if (ret < 0 || type != DAV1D_OBU_TD || obu_size > 0)
        return 0;
    cnt += ret;

    // look for first frame and accompanying sequence header
    int seq = 0;
    while (cnt < PROBE_SIZE) {
        ret = parse_obu_header(data + cnt, PROBE_SIZE - cnt,
                               &obu_size, &type, 0);
        if (ret < 0)
            return 0;
        cnt += ret;

        switch (type) {
        case DAV1D_OBU_SEQ_HDR:
            seq = 1;
            break;
        case DAV1D_OBU_FRAME:
        case DAV1D_OBU_FRAME_HDR:
            return seq;
        case DAV1D_OBU_TD:
        case DAV1D_OBU_TILE_GRP:
            return 0;
        default:
            break;
        }
    }

    return 0;
}

typedef struct DemuxerPriv {
    FILE *f;
} Section5InputContext;

static int section5_open(Section5InputContext *const c, const char *const file,
                         unsigned fps[2], unsigned *const num_frames, unsigned timebase[2])
{
    if (!(c->f = fopen(file, "rb"))) {
        fprintf(stderr, "Failed to open %s: %s\n", file, strerror(errno));
        return -1;
    }

    // TODO: Parse sequence header and read timing info if any.
    fps[0] = 25;
    fps[1] = 1;
    timebase[0] = 25;
    timebase[1] = 1;
    *num_frames = 0;
    for (;;) {
        uint8_t byte[2];

        if (fread(&byte[0], 1, 1, c->f) < 1)
            break;
        const enum Dav1dObuType obu_type = (byte[0] >> 3) & 0xf;
        if (obu_type == DAV1D_OBU_TD)
            (*num_frames)++;
        const int has_length_field = byte[0] & 0x2;
        if (!has_length_field)
            return -1;
        const int has_extension = byte[0] & 0x4;
        if (has_extension && fread(&byte[1], 1, 1, c->f) < 1)
            return -1;
        size_t len;
        const int res = leb128(c->f, &len);
        if (res < 0)
            return -1;
        fseeko(c->f, len, SEEK_CUR); // skip packet
    }
    fseeko(c->f, 0, SEEK_SET);

    return 0;
}

static int section5_read(Section5InputContext *const c, Dav1dData *const data) {
    size_t total_bytes = 0;

    for (int first = 1;; first = 0) {
        uint8_t byte[2];

        if (fread(&byte[0], 1, 1, c->f) < 1) {
            if (!first && feof(c->f)) break;
            return -1;
        }
        const enum Dav1dObuType obu_type = (byte[0] >> 3) & 0xf;
        if (first) {
            if (obu_type != DAV1D_OBU_TD)
                return -1;
        } else {
            if (obu_type == DAV1D_OBU_TD) {
                // include TD in next packet
                fseeko(c->f, -1, SEEK_CUR);
                break;
            }
        }
        const int has_length_field = byte[0] & 0x2;
        if (!has_length_field)
            return -1;
        const int has_extension = !!(byte[0] & 0x4);
        if (has_extension && fread(&byte[1], 1, 1, c->f) < 1)
            return -1;
        size_t len;
        const int res = leb128(c->f, &len);
        if (res < 0)
            return -1;
        total_bytes += 1 + has_extension + res + len;
        fseeko(c->f, len, SEEK_CUR); // skip packet, we'll read it below
    }

    fseeko(c->f, -(off_t)total_bytes, SEEK_CUR);
    uint8_t *ptr = dav1d_data_create(data, total_bytes);
    if (!ptr) return -1;
    if (fread(ptr, total_bytes, 1, c->f) != 1) {
        fprintf(stderr, "Failed to read frame data: %s\n", strerror(errno));
        dav1d_data_unref(data);
        return -1;
    }

    return 0;
}

static void section5_close(Section5InputContext *const c) {
    fclose(c->f);
}

const Demuxer section5_demuxer = {
    .priv_data_size = sizeof(Section5InputContext),
    .name = "section5",
    .probe = section5_probe,
    .probe_sz = PROBE_SIZE,
    .open = section5_open,
    .read = section5_read,
    .seek = NULL,
    .close = section5_close,
};
