/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 20, 2023.
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
#include "krb5_locl.h"
#include "store-int.h"

typedef struct mem_storage{
    unsigned char *base;
    size_t size;
    unsigned char *ptr;
}mem_storage;

static ssize_t
mem_fetch(krb5_storage *sp, void *data, size_t size)
{
    mem_storage *s = (mem_storage*)sp->data;
    if(size > (size_t)(s->base + s->size - s->ptr))
	size = s->base + s->size - s->ptr;
    memmove(data, s->ptr, size);
    sp->seek(sp, size, SEEK_CUR);
    return size;
}

static ssize_t
mem_store(krb5_storage *sp, const void *data, size_t size)
{
    mem_storage *s = (mem_storage*)sp->data;
    if(size > (size_t)(s->base + s->size - s->ptr))
	size = s->base + s->size - s->ptr;
    memmove(s->ptr, data, size);
    sp->seek(sp, size, SEEK_CUR);
    return size;
}

static ssize_t
mem_no_store(krb5_storage *sp, const void *data, size_t size)
{
    return -1;
}

static off_t
mem_seek(krb5_storage *sp, off_t offset, int whence)
{
    mem_storage *s = (mem_storage*)sp->data;
    switch(whence){
    case SEEK_SET:
	if((size_t)offset > s->size)
	    offset = s->size;
	if(offset < 0)
	    offset = 0;
	s->ptr = s->base + offset;
	break;
    case SEEK_CUR:
	return sp->seek(sp, s->ptr - s->base + offset, SEEK_SET);
    case SEEK_END:
	return sp->seek(sp, s->size + offset, SEEK_SET);
    default:
	errno = EINVAL;
	return -1;
    }
    return s->ptr - s->base;
}

static int
mem_trunc(krb5_storage *sp, off_t offset)
{
    mem_storage *s = (mem_storage*)sp->data;
    /* make SIZE_T_MAX to something that wont turn into -1 */
    if(offset > (off_t)(SIZE_T_MAX >> 1) || (size_t)offset > s->size)
	return ERANGE;
    s->size = (size_t)offset;
    if ((s->ptr - s->base) > offset)
	s->ptr = s->base + offset;
    return 0;
}

static int
mem_no_trunc(krb5_storage *sp, off_t offset)
{
    return EINVAL;
}

static void
mem_free(krb5_storage *sp)
{
    mem_storage *s = sp->data;
    memset(s->base, 0, s->size);
    free(s->base);
}


/**
 * Create a fixed size memory storage block
 *
 * @return A krb5_storage on success, or NULL on out of memory error.
 *
 * @ingroup krb5_storage
 *
 * @sa krb5_storage_emem()
 * @sa krb5_storage_from_readonly_mem()
 * @sa krb5_storage_from_data()
 * @sa krb5_storage_from_fd()
 * @sa krb5_storage_from_mem_copy()
 */

KRB5_LIB_FUNCTION krb5_storage * KRB5_LIB_CALL
krb5_storage_from_mem(void *buf, size_t len)
{
    krb5_storage *sp = malloc(sizeof(krb5_storage));
    mem_storage *s;
    if(sp == NULL)
	return NULL;
    s = malloc(sizeof(*s));
    if(s == NULL) {
	free(sp);
	return NULL;
    }
    sp->data = s;
    sp->flags = 0;
    sp->eof_code = HEIM_ERR_EOF;
    s->base = buf;
    s->size = len;
    s->ptr = buf;
    sp->fetch = mem_fetch;
    sp->store = mem_store;
    sp->seek = mem_seek;
    sp->trunc = mem_trunc;
    sp->free = NULL;
    sp->max_alloc = UINT_MAX/8;
    return sp;
}

/**
 * Create a fixed size memory storage block
 *
 * @return A krb5_storage on success, or NULL on out of memory error.
 *
 * @ingroup krb5_storage
 *
 * @sa krb5_storage_mem()
 * @sa krb5_storage_from_mem()
 * @sa krb5_storage_from_readonly_mem()
 * @sa krb5_storage_from_fd()
 * @sa krb5_storage_from_mem_copy()
 */

KRB5_LIB_FUNCTION krb5_storage * KRB5_LIB_CALL
krb5_storage_from_data(krb5_data *data)
{
    return krb5_storage_from_mem(data->data, data->length);
}

/**
 * Create a fixed size memory storage block that is read only
 *
 * @return A krb5_storage on success, or NULL on out of memory error.
 *
 * @ingroup krb5_storage
 *
 * @sa krb5_storage_emem()
 * @sa krb5_storage_from_mem()
 * @sa krb5_storage_from_data()
 * @sa krb5_storage_from_fd()
 * @sa krb5_storage_from_mem_copy()
 */

KRB5_LIB_FUNCTION krb5_storage * KRB5_LIB_CALL
krb5_storage_from_readonly_mem(const void *buf, size_t len)
{
    krb5_storage *sp = malloc(sizeof(krb5_storage));
    mem_storage *s;
    if(sp == NULL)
	return NULL;
    s = malloc(sizeof(*s));
    if(s == NULL) {
	free(sp);
	return NULL;
    }
    sp->data = s;
    sp->flags = 0;
    sp->eof_code = HEIM_ERR_EOF;
    s->base = rk_UNCONST(buf);
    s->size = len;
    s->ptr = rk_UNCONST(buf);
    sp->fetch = mem_fetch;
    sp->store = mem_no_store;
    sp->seek = mem_seek;
    sp->trunc = mem_no_trunc;
    sp->free = NULL;
    sp->max_alloc = UINT_MAX/8;
    return sp;
}

/**
 * Create a copy of a memory and assign it to a storage block
 *
 * The input data buffer is copied and the orignal buffer can be freed
 * during the life the storage.
 *
 * @return A krb5_storage on success, or NULL on out of memory error.
 *
 * @ingroup krb5_storage
 *
 * @sa krb5_storage_emem()
 * @sa krb5_storage_from_mem()
 * @sa krb5_storage_from_data()
 * @sa krb5_storage_from_fd()
 */

krb5_storage * KRB5_LIB_FUNCTION
krb5_storage_from_mem_copy(void *buf, size_t len)
{
    krb5_storage *sp = malloc(sizeof(krb5_storage));
    mem_storage *s;
    if(sp == NULL)
	return NULL;
    s = malloc(sizeof(*s));
    if(s == NULL) {
	free(sp);
	return NULL;
    }
    sp->data = s;
    sp->flags = 0;
    sp->eof_code = HEIM_ERR_EOF;
    s->base = malloc(len);
    if (s->base == NULL) {
	free(sp);
	free(s);
	return NULL;
    }
    memcpy(s->base, buf, len);
    s->size = len;
    s->ptr = s->base;
    sp->fetch = mem_fetch;
    sp->store = mem_store;
    sp->seek = mem_seek;
    sp->trunc = mem_trunc;
    sp->free = mem_free;
    return sp;
}
