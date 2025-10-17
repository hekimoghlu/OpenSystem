/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 18, 2025.
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
#ifndef DLZ_LIST_H
#define DLZ_LIST_H 1

#define DLZ_LIST(type) struct { type *head, *tail; }
#define DLZ_LIST_INIT(list) \
	do { (list).head = NULL; (list).tail = NULL; } while (0)

#define DLZ_LINK(type) struct { type *prev, *next; }
#define DLZ_LINK_INIT(elt, link) \
	do { \
		(elt)->link.prev = (void *)(-1); \
		(elt)->link.next = (void *)(-1); \
	} while (0)

#define DLZ_LIST_HEAD(list) ((list).head)
#define DLZ_LIST_TAIL(list) ((list).tail)

#define DLZ_LIST_APPEND(list, elt, link) \
	do { \
		if ((list).tail != NULL) \
			(list).tail->link.next = (elt); \
		else \
			(list).head = (elt); \
		(elt)->link.prev = (list).tail; \
		(elt)->link.next = NULL; \
		(list).tail = (elt); \
	} while (0)

#define DLZ_LIST_PREV(elt, link) ((elt)->link.prev)
#define DLZ_LIST_NEXT(elt, link) ((elt)->link.next)

#endif /* DLZ_LIST_H */
