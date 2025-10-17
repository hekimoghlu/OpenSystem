/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 16, 2024.
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
 * Copyright 2003 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

#ifndef	_DT_LIST_H
#define	_DT_LIST_H

#ifdef	__cplusplus
extern "C" {
#endif

typedef struct dt_list {
	struct dt_list *dl_prev;
	struct dt_list *dl_next;
} dt_list_t;

#define	dt_list_prev(elem)	((void *)(((dt_list_t *)(elem))->dl_prev))
#define	dt_list_next(elem)	((void *)(((dt_list_t *)(elem))->dl_next))

extern void dt_list_append(dt_list_t *, void *);
extern void dt_list_prepend(dt_list_t *, void *);
extern void dt_list_insert(dt_list_t *, void *, void *);
extern void dt_list_delete(dt_list_t *, void *);

#ifdef	__cplusplus
}
#endif

#endif	/* _DT_LIST_H */
