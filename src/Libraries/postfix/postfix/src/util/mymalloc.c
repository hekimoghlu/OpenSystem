/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 9, 2022.
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
/* System libraries. */

#include "sys_defs.h"
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

/* Application-specific. */

#include "msg.h"
#include "mymalloc.h"

 /*
  * Structure of an annotated memory block. In order to detect spurious
  * free() calls we prepend a signature to memory given to the application.
  * In order to detect access to free()d blocks, overwrite each block as soon
  * as it is passed to myfree(). With the code below, the user data has
  * integer alignment or better.
  */
typedef struct MBLOCK {
    int     signature;			/* set when block is active */
    ssize_t length;			/* user requested length */
    union {
	ALIGN_TYPE align;
	char    payload[1];		/* actually a bunch of bytes */
    }       u;
} MBLOCK;

#define SIGNATURE	0xdead
#define FILLER		0xff

#define CHECK_IN_PTR(ptr, real_ptr, len, fname) { \
    if (ptr == 0) \
	msg_panic("%s: null pointer input", fname); \
    real_ptr = (MBLOCK *) (ptr - offsetof(MBLOCK, u.payload[0])); \
    if (real_ptr->signature != SIGNATURE) \
	msg_panic("%s: corrupt or unallocated memory block", fname); \
    real_ptr->signature = 0; \
    if ((len = real_ptr->length) < 1) \
	msg_panic("%s: corrupt memory block length", fname); \
}

#define CHECK_OUT_PTR(ptr, real_ptr, len) { \
    real_ptr->signature = SIGNATURE; \
    real_ptr->length = len; \
    ptr = real_ptr->u.payload; \
}

#define SPACE_FOR(len)	(offsetof(MBLOCK, u.payload[0]) + len)

 /*
  * Optimization for short strings. We share one copy with multiple callers.
  * This differs from normal heap memory in two ways, because the memory is
  * shared:
  * 
  * -  It must be read-only to avoid horrible bugs. This is OK because there is
  * no legitimate reason to modify the null terminator.
  * 
  * - myfree() cannot overwrite the memory with a filler pattern like it can do
  * with heap memory. Therefore, some dangling pointer bugs will be masked.
  */
#ifndef NO_SHARED_EMPTY_STRINGS
static const char empty_string[] = "";

#endif

/* mymalloc - allocate memory or bust */

void   *mymalloc(ssize_t len)
{
    void   *ptr;
    MBLOCK *real_ptr;

    /*
     * Note: for safety reasons the request length is a signed type. This
     * allows us to catch integer overflow problems that weren't already
     * caught up-stream.
     */
    if (len < 1)
	msg_panic("mymalloc: requested length %ld", (long) len);
#ifdef MYMALLOC_FUZZ
    len += MYMALLOC_FUZZ;
#endif
    if ((real_ptr = (MBLOCK *) malloc(SPACE_FOR(len))) == 0)
	msg_fatal("mymalloc: insufficient memory for %ld bytes: %m",
		  (long) len);
    CHECK_OUT_PTR(ptr, real_ptr, len);
    memset(ptr, FILLER, len);
    return (ptr);
}

/* myrealloc - reallocate memory or bust */

void   *myrealloc(void *ptr, ssize_t len)
{
    MBLOCK *real_ptr;
    ssize_t old_len;

#ifndef NO_SHARED_EMPTY_STRINGS
    if (ptr == empty_string)
	return (mymalloc(len));
#endif

    /*
     * Note: for safety reasons the request length is a signed type. This
     * allows us to catch integer overflow problems that weren't already
     * caught up-stream.
     */
    if (len < 1)
	msg_panic("myrealloc: requested length %ld", (long) len);
#ifdef MYMALLOC_FUZZ
    len += MYMALLOC_FUZZ;
#endif
    CHECK_IN_PTR(ptr, real_ptr, old_len, "myrealloc");
    if ((real_ptr = (MBLOCK *) realloc((void *) real_ptr, SPACE_FOR(len))) == 0)
	msg_fatal("myrealloc: insufficient memory for %ld bytes: %m",
		  (long) len);
    CHECK_OUT_PTR(ptr, real_ptr, len);
    if (len > old_len)
	memset(ptr + old_len, FILLER, len - old_len);
    return (ptr);
}

/* myfree - release memory */

void    myfree(void *ptr)
{
    MBLOCK *real_ptr;
    ssize_t len;

#ifndef NO_SHARED_EMPTY_STRINGS
    if (ptr != empty_string) {
#endif
	CHECK_IN_PTR(ptr, real_ptr, len, "myfree");
	memset((void *) real_ptr, FILLER, SPACE_FOR(len));
	free((void *) real_ptr);
#ifndef NO_SHARED_EMPTY_STRINGS
    }
#endif
}

/* mystrdup - save string to heap */

char   *mystrdup(const char *str)
{
    size_t  len;

    if (str == 0)
	msg_panic("mystrdup: null pointer argument");
#ifndef NO_SHARED_EMPTY_STRINGS
    if (*str == 0)
	return ((char *) empty_string);
#endif
    if ((len = strlen(str) + 1) > SSIZE_T_MAX)
	msg_panic("mystrdup: string length >= SSIZE_T_MAX");
    return (strcpy(mymalloc(len), str));
}

/* mystrndup - save substring to heap */

char   *mystrndup(const char *str, ssize_t len)
{
    char   *result;
    char   *cp;

    if (str == 0)
	msg_panic("mystrndup: null pointer argument");
    if (len < 0)
	msg_panic("mystrndup: requested length %ld", (long) len);
#ifndef NO_SHARED_EMPTY_STRINGS
    if (*str == 0)
	return ((char *) empty_string);
#endif
    if ((cp = memchr(str, 0, len)) != 0)
	len = cp - str;
    result = memcpy(mymalloc(len + 1), str, len);
    result[len] = 0;
    return (result);
}

/* mymemdup - copy memory */

char   *mymemdup(const void *ptr, ssize_t len)
{
    if (ptr == 0)
	msg_panic("mymemdup: null pointer argument");
    return (memcpy(mymalloc(len), ptr, len));
}
