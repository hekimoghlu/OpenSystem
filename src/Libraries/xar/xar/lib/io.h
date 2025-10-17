/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 18, 2024.
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
/*
 * 03-Apr-2005
 * DRI: Rob Braun <bbraun@synack.net>
 */
/*
 * Portions Copyright 2006, Apple Computer, Inc.
 * Christopher Ryan <ryanc@apple.com>
*/

#ifndef _XAR_IO_H_
#define _XAR_IO_H_

#include "archive.h"
//typedef int (*read_callback)(xar_t, xar_file_t, void *, size_t, void *context);
//typedef int (*write_callback)(xar_t, xar_file_t, void *, size_t, void *context);

typedef int (*fromheap_in)(xar_t x, xar_file_t f, xar_prop_t p, void **in, size_t *inlen, void **context);
typedef int (*fromheap_out)(xar_t x, xar_file_t f, xar_prop_t p, void *in, size_t inlen, void **context);
typedef int (*fromheap_done)(xar_t x, xar_file_t f, xar_prop_t p, void **context);

typedef int (*toheap_in)(xar_t x, xar_file_t f, xar_prop_t p, void **in, size_t *inlen, void **context);
typedef int (*toheap_out)(xar_t x, xar_file_t f, xar_prop_t p, void *in, size_t inlen, void **context);
typedef int (*toheap_done)(xar_t x, xar_file_t f, xar_prop_t p, void **context);

struct datamod {
	fromheap_in      fh_in;
	fromheap_out     fh_out;
	fromheap_done    fh_done;
	toheap_in        th_in;
	toheap_out       th_out;
	toheap_done      th_done;
};

typedef struct xar_stream_state {
        char      *pending_buf;
        size_t     pending_buf_size;

        void     **modulecontext;
        int        modulecount;
        size_t     bsize;
        int64_t    fsize;
        xar_t      x;
        xar_file_t f;
	xar_prop_t p;
} xar_stream_state_t;

size_t xar_io_get_rsize(xar_t x);
off_t xar_io_get_heap_base_offset(xar_t x);
size_t xar_io_get_toc_checksum_length_for_type(const char *type);
size_t xar_io_get_toc_checksum_length(xar_t x);
off_t xar_io_get_file_offset(xar_t x, xar_file_t f, xar_prop_t p);
int64_t xar_io_get_length(xar_prop_t p);

int32_t xar_attrcopy_to_heap(xar_t x, xar_file_t f, xar_prop_t p, read_callback rcb, void *context);
int32_t xar_attrcopy_from_heap(xar_t x, xar_file_t f, xar_prop_t p, write_callback wcb, void *context);
int32_t xar_attrcopy_from_heap_to_heap(xar_t xsource, xar_file_t fsource, xar_prop_t p, xar_t xdest, xar_file_t fdest);
int32_t xar_attrcopy_from_heap_to_stream_init(xar_t x, xar_file_t f, xar_prop_t p, xar_stream *stream);
int32_t xar_attrcopy_from_heap_to_stream(xar_stream *stream);
int32_t xar_attrcopy_from_heap_to_stream_end(xar_stream *stream);

int32_t xar_heap_to_archive(xar_t x);

#pragma mark internal

// IMPORTANT: Keep count synchronized with declaration in io.c!
extern struct datamod xar_datamods[6];

#endif /* _XAR_IO_H_ */
