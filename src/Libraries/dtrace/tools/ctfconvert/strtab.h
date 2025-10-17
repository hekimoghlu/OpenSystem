/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 2, 2025.
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
 * Copyright (c) 2001 by Sun Microsystems, Inc.
 * All rights reserved.
 */

#ifndef	_STRTAB_H
#define	_STRTAB_H

#include <sys/types.h>

#include "atom.h"

#ifdef	__cplusplus
extern "C" {
#endif

typedef struct strhash strhash_t;

typedef struct strtab {
	strhash_t *str_hash;		/* array of hash buckets */
	char **str_bufs;		/* array of buffer pointers */
	char *str_ptr;			/* pointer to current buffer location */
	ulong_t str_nbufs;		/* size of buffer pointer array */
	size_t str_bufsz;		/* size of individual buffer */
	size_t str_size;		/* total size of strings in bytes */
} strtab_t;

extern void strtab_create(strtab_t *);
extern void strtab_destroy(strtab_t *);
extern size_t strtab_insert(strtab_t *, atom_t *);
extern size_t strtab_size(const strtab_t *);
extern ssize_t strtab_write(const strtab_t *,
    ssize_t (*)(const void *, size_t, void *), void *);

#ifdef	__cplusplus
}
#endif

#endif	/* _STRTAB_H */
