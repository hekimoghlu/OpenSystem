/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 11, 2025.
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
/* $Id: list.h,v 1.14 2007/06/19 23:47:23 tbox Exp $ */

#ifndef LWRES_LIST_H
#define LWRES_LIST_H 1

/*! \file lwres/list.h */

#define LWRES_LIST(type) struct { type *head, *tail; }
#define LWRES_LIST_INIT(list) \
	do { (list).head = NULL; (list).tail = NULL; } while (0)

#define LWRES_LINK(type) struct { type *prev, *next; }
#define LWRES_LINK_INIT(elt, link) \
	do { \
		(elt)->link.prev = (void *)(-1); \
		(elt)->link.next = (void *)(-1); \
	} while (0)
#define LWRES_LINK_LINKED(elt, link) \
	((void *)((elt)->link.prev) != (void *)(-1))

#define LWRES_LIST_HEAD(list) ((list).head)
#define LWRES_LIST_TAIL(list) ((list).tail)
#define LWRES_LIST_EMPTY(list) LWRES_TF((list).head == NULL)

#define LWRES_LIST_PREPEND(list, elt, link) \
	do { \
		if ((list).head != NULL) \
			(list).head->link.prev = (elt); \
		else \
			(list).tail = (elt); \
		(elt)->link.prev = NULL; \
		(elt)->link.next = (list).head; \
		(list).head = (elt); \
	} while (0)

#define LWRES_LIST_APPEND(list, elt, link) \
	do { \
		if ((list).tail != NULL) \
			(list).tail->link.next = (elt); \
		else \
			(list).head = (elt); \
		(elt)->link.prev = (list).tail; \
		(elt)->link.next = NULL; \
		(list).tail = (elt); \
	} while (0)

#define LWRES_LIST_UNLINK(list, elt, link) \
	do { \
		if ((elt)->link.next != NULL) \
			(elt)->link.next->link.prev = (elt)->link.prev; \
		else \
			(list).tail = (elt)->link.prev; \
		if ((elt)->link.prev != NULL) \
			(elt)->link.prev->link.next = (elt)->link.next; \
		else \
			(list).head = (elt)->link.next; \
		(elt)->link.prev = (void *)(-1); \
		(elt)->link.next = (void *)(-1); \
	} while (0)

#define LWRES_LIST_PREV(elt, link) ((elt)->link.prev)
#define LWRES_LIST_NEXT(elt, link) ((elt)->link.next)

#define LWRES_LIST_INSERTBEFORE(list, before, elt, link) \
	do { \
		if ((before)->link.prev == NULL) \
			LWRES_LIST_PREPEND(list, elt, link); \
		else { \
			(elt)->link.prev = (before)->link.prev; \
			(before)->link.prev = (elt); \
			(elt)->link.prev->link.next = (elt); \
			(elt)->link.next = (before); \
		} \
	} while (0)

#define LWRES_LIST_INSERTAFTER(list, after, elt, link) \
	do { \
		if ((after)->link.next == NULL) \
			LWRES_LIST_APPEND(list, elt, link); \
		else { \
			(elt)->link.next = (after)->link.next; \
			(after)->link.next = (elt); \
			(elt)->link.next->link.prev = (elt); \
			(elt)->link.prev = (after); \
		} \
	} while (0)

#define LWRES_LIST_APPENDLIST(list1, list2, link) \
	do { \
		if (LWRES_LIST_EMPTY(list1)) \
			(list1) = (list2); \
		else if (!LWRES_LIST_EMPTY(list2)) { \
			(list1).tail->link.next = (list2).head; \
			(list2).head->link.prev = (list1).tail; \
			(list1).tail = (list2).tail; \
		} \
		(list2).head = NULL; \
		(list2).tail = NULL; \
	} while (0)

#define LWRES_LIST_ENQUEUE(list, elt, link) LWRES_LIST_APPEND(list, elt, link)
#define LWRES_LIST_DEQUEUE(list, elt, link) LWRES_LIST_UNLINK(list, elt, link)

#endif /* LWRES_LIST_H */
