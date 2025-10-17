/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 19, 2023.
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
 * Copyright 2001-2003 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

#ifndef _ASSOC_H
#define	_ASSOC_H

/*
 * Create, manage, and destroy association lists.  alists are arrays with
 * arbitrary index types.
 */

#ifdef __cplusplus
extern "C" {
#endif

#define	ALIST_HASH_SIZE	128

typedef struct alist alist_t;

alist_t *alist_new(unsigned size);
void alist_clear(alist_t *);
void alist_free(alist_t *);
void alist_add(alist_t *, void *, void *);
int alist_find(alist_t *, void *, void **);
int alist_iter(alist_t *, int (*)(void *, void *, void *), void *);
void alist_stats(alist_t *, int);

#ifdef __cplusplus
}
#endif

#endif /* _ASSOC_H */
