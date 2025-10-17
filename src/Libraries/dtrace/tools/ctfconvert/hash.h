/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 19, 2023.
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
 * Copyright 2004 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

#ifndef _HASH_H
#define	_HASH_H

/*
 * Routines for manipulating hash tables
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hash hash_t;

hash_t *hash_new(int, int (*)(int, void *), int (*)(void *, void *));
void hash_add(hash_t *, void *);
void hash_merge(hash_t *, hash_t *);
void hash_remove(hash_t *, void *);
int hash_find(hash_t *, void *, void **);
int hash_find_iter(hash_t *, void *, int (*)(void *, void *), void *);
int hash_iter(hash_t *, int (*)(void *, void *), void *);
int hash_match(hash_t *, void *, int (*)(void *, void *), void *);
int hash_count(hash_t *);
int hash_name(int, const char *);
void hash_stats(hash_t *, int);
void hash_free(hash_t *, void (*)(void *, void *), void *);

#ifdef __cplusplus
}
#endif

#endif /* _HASH_H */
