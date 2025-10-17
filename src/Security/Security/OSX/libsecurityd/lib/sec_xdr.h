/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 17, 2024.
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
#ifndef _SEC_XDR_H
#define _SEC_XDR_H

#include <rpc/types.h>
#include <rpc/xdr.h>
#include <stdint.h>
 
__BEGIN_DECLS

extern bool_t sec_xdr_bytes(XDR *, uint8_t **, unsigned int *, unsigned int);
extern bool_t sec_xdr_array(XDR *, uint8_t **, unsigned int *, unsigned int, unsigned int, xdrproc_t);
extern bool_t sec_xdr_charp(XDR *, char **, u_int);
extern bool_t sec_xdr_reference(XDR *xdrs, uint8_t **pp, u_int size, xdrproc_t proc);
extern bool_t sec_xdr_pointer(XDR *xdrs, uint8_t **objpp, u_int obj_size, xdrproc_t xdr_obj);

bool_t sec_mem_alloc(XDR *xdr, u_int bsize, uint8_t **data);
void sec_mem_free(XDR *xdr, void *ptr, u_int bsize);

void sec_xdrmem_create(XDR *xdrs, char *addr, u_int size, enum xdr_op op);

typedef struct sec_xdr_arena_allocator {
    uint32_t magic;
    uint8_t *offset;
    uint8_t *data;
    uint8_t *end;
} sec_xdr_arena_allocator_t;

#define xdr_arena_magic 0xAEA1
#define xdr_size_magic 0xDEAD

void sec_xdr_arena_init_size_alloc(sec_xdr_arena_allocator_t *arena, XDR *xdr);
bool_t sec_xdr_arena_init(sec_xdr_arena_allocator_t *arena, XDR *xdr,
                size_t in_length, uint8_t *in_data);
void sec_xdr_arena_free(sec_xdr_arena_allocator_t *alloc, void *ptr, size_t bsize);
void *sec_xdr_arena_data(sec_xdr_arena_allocator_t *alloc);
sec_xdr_arena_allocator_t *sec_xdr_arena_allocator(XDR *xdr);
bool_t sec_xdr_arena_size_allocator(XDR *xdr);

bool_t copyin(void * data, xdrproc_t proc, void ** copy, u_int * size);
bool_t copyout(const void * copy, u_int size, xdrproc_t proc, void ** data, u_int *length);
bool_t copyout_chunked(const void * copy, u_int size, xdrproc_t proc, void ** data);

u_int sec_xdr_sizeof_in(xdrproc_t func, void * data);
u_int sec_xdr_sizeof_out(const void * copy, u_int size, xdrproc_t func, void ** data);

__END_DECLS

#endif /* !_SEC_XDR_H */
