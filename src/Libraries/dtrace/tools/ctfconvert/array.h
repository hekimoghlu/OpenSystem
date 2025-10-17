/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 20, 2023.
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
#ifndef _ARRAY_H
#define	_ARRAY_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct array array_t;

int array_count(const array_t *);
void *array_get(const array_t *, int);

void array_add(array_t **, void *);
void array_concat(array_t **, array_t **);
void array_clear(array_t *, void (*)(void *, void *), void *);
void array_free(array_t **, void (*)(void *, void *), void *);

int array_iter(const array_t *, int (*)(void *, void *), void *);
#define ARRAY_ABORT  -1
#define ARRAY_KEEP   0
#define ARRAY_REMOVE 1
int array_filter(array_t *, int (*)(void *, void *), void *);
void array_sort(array_t *, int (*)(void *, void *));

#ifdef __cplusplus
}
#endif

#endif /* _ARRAY_H */
