/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 16, 2024.
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

#include <Foundation/Foundation.h>

#include "config.h"
#include "array.c"
#include "ipp-support.c"
#include "options.c"
#include "transcode.c"
#include "usersys.c"
#include "language.c"
#include "thread.c"
#include "ipp.c"
#include "globals.c"
#include "debug.c"
#include "file.c"
#include "hash.c"
#include "dir.c"

#include "stubs.m"

ipp_state_t                /* O - Current state */
ippWriteIO2(void           *dst,        /* I - Destination */
			ipp_iocb_t cb,        /* I - Write callback function */
			ipp_t          *parent,        /* I - Parent IPP message */
			ipp_t          *ipp);        /* I - IPP data */


struct Payload {
    size_t        pos;
    size_t        len;
    ipp_uchar_t*  buf;
};

static ssize_t _ipp_read_cb(void* context, ipp_uchar_t* buffer, size_t bytes)
{
    struct Payload* p = (struct Payload*) context;

    if ((p->pos + bytes) > p->len) {
        if (gVerbose) {
            NSLog(@"%s: attempt to read %ld bytes at %ld > %ld length\n", __FUNCTION__, bytes, p->pos, p->len);
        }
        return -1;
    }

    memcpy(buffer, &p->buf[p->pos], bytes);
    p->pos += bytes;

    return (ssize_t) bytes;
}

static ssize_t _ipp_write_cb(void* context, ipp_uchar_t* buffer, size_t bytes)
{
    struct Payload* p = (struct Payload*) context;

    p->pos = p->len;
    p->buf = realloc(p->buf, p->pos + bytes);
    memcpy(&p->buf[p->pos], buffer, bytes);
    p->len += bytes;

    return (ssize_t) bytes;
}

static unsigned long long md5(const ipp_uchar_t* buffer, size_t bytes)
{
    union {
        unsigned long long result;
        ipp_uchar_t tmp[64];
    } tmp;

    bzero(&tmp, sizeof(tmp));

    cupsHashData("md5", (const void*) buffer, bytes, &tmp.tmp[0], sizeof(tmp.tmp));

    return tmp.result;
}

static void failErr(const char* file, const char* msg, int err)
{
    NSLog(@"Error: %s for %s (%d %s)\n", msg, file, err, strerror(err));
    exit(-1);
}

static void fuzz0(ipp_uchar_t* p, size_t len, int first_pass, char* outbuf, size_t outbufLen)
{
    struct Payload r = {
        0,
        len,
        p
    };

    struct Payload w = {
        0,
        0,
        NULL
    };

    ipp_t* job = ippNew();

    if (ippReadIO(&r, _ipp_read_cb, 1, NULL, job) < IPP_STATE_IDLE) {
        snprintf(outbuf, outbufLen, "ERR couldn't read into ipp");
    } else {
        ippSetState(job, IPP_STATE_IDLE);

		ipp_state_t wstate = ippWriteIO(&w, _ipp_write_cb, 1, NULL, job);

		if (wstate >= IPP_STATE_IDLE) {
			struct Payload w2 = {
				0,
				0,
				NULL
			};
			ippSetState(job, IPP_STATE_IDLE);

			ipp_state_t w2state = ippWriteIO2(&w2, _ipp_write_cb, NULL, job);
			assert(w2state = wstate);
			assert(w2.len == w.len);
			assert(memcmp(w2.buf, w.buf, w.len) == 0);
		}

        if (wstate < IPP_STATE_IDLE) {
            snprintf(outbuf, outbufLen, "ERR couldn't write from ipp");
        } else {
            int pos = snprintf(outbuf, outbufLen, "ERR read %ld(%llx) write %ld(%llx)", r.len, md5(r.buf, r.len), w.len, md5(w.buf, w.len));

            if (first_pass) {
                char secondBuf[1024];

                fuzz0(w.buf, w.len, 0, secondBuf, sizeof(secondBuf));

                int mismatch = (strcmp(outbuf, secondBuf) != 0);

                assert((int) outbufLen > pos);
                snprintf(&outbuf[pos], outbufLen - (size_t) pos, " / vs %s%s", secondBuf, mismatch? " (ERR mismatch)" : "");
            }
        }
    }

    ippDelete(job);

    free((char*) w.buf);
}

int _ipp_fuzzing(Boolean verbose, const uint8_t* data, size_t len)
{
    Boolean save = gVerbose;
    gVerbose = verbose;

    char outbuf[1024] = { 0 };

    fuzz0((ipp_uchar_t*) data, len, 1, outbuf, sizeof(outbuf));

    if (gVerbose) {
        NSLog(@"%s", outbuf);
    }

    gVerbose = save;
    
    return strstr(outbuf, "ERR") == nil;
}

extern int LLVMFuzzerTestOneInput(const uint8_t *buffer, size_t size);

int LLVMFuzzerTestOneInput(const uint8_t *buffer, size_t size)
{
    return _ipp_fuzzing(gVerbose, buffer, size);
}

