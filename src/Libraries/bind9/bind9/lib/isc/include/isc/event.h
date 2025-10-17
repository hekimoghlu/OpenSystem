/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 6, 2022.
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
#ifndef ISC_EVENT_H
#define ISC_EVENT_H 1

/*! \file isc/event.h */

#include <isc/lang.h>
#include <isc/types.h>

/*****
 ***** Events.
 *****/

typedef void (*isc_eventdestructor_t)(isc_event_t *);

#define ISC_EVENT_COMMON(ltype)		\
	size_t				ev_size; \
	unsigned int			ev_attributes; \
	void *				ev_tag; \
	isc_eventtype_t			ev_type; \
	isc_taskaction_t		ev_action; \
	void *				ev_arg; \
	void *				ev_sender; \
	isc_eventdestructor_t		ev_destroy; \
	void *				ev_destroy_arg; \
	ISC_LINK(ltype)			ev_link; \
	ISC_LINK(ltype)			ev_ratelink

/*%
 * Attributes matching a mask of 0x000000ff are reserved for the task library's
 * definition.  Attributes of 0xffffff00 may be used by the application
 * or non-ISC libraries.
 */
#define ISC_EVENTATTR_NOPURGE		0x00000001

/*%
 * The ISC_EVENTATTR_CANCELED attribute is intended to indicate
 * that an event is delivered as a result of a canceled operation
 * rather than successful completion, by mutual agreement
 * between the sender and receiver.  It is not set or used by
 * the task system.
 */
#define ISC_EVENTATTR_CANCELED		0x00000002

#define ISC_EVENT_INIT(event, sz, at, ta, ty, ac, ar, sn, df, da) \
do { \
	(event)->ev_size = (sz); \
	(event)->ev_attributes = (at); \
	(event)->ev_tag = (ta); \
	(event)->ev_type = (ty); \
	(event)->ev_action = (ac); \
	(event)->ev_arg = (ar); \
	(event)->ev_sender = (sn); \
	(event)->ev_destroy = (df); \
	(event)->ev_destroy_arg = (da); \
	ISC_LINK_INIT((event), ev_link); \
	ISC_LINK_INIT((event), ev_ratelink); \
} while (0)

/*%
 * This structure is public because "subclassing" it may be useful when
 * defining new event types.
 */
struct isc_event {
	ISC_EVENT_COMMON(struct isc_event);
};

#define ISC_EVENTTYPE_FIRSTEVENT	0x00000000
#define ISC_EVENTTYPE_LASTEVENT		0xffffffff

#define ISC_EVENT_PTR(p) ((isc_event_t **)(void *)(p))

ISC_LANG_BEGINDECLS

isc_event_t *
isc_event_allocate(isc_mem_t *mctx, void *sender, isc_eventtype_t type,
		   isc_taskaction_t action, void *arg, size_t size);
isc_event_t *
isc_event_constallocate(isc_mem_t *mctx, void *sender, isc_eventtype_t type,
			isc_taskaction_t action, const void *arg, size_t size);
/*%<
 * Allocate an event structure.
 *
 * Allocate and initialize in a structure with initial elements
 * defined by:
 *
 * \code
 *	struct {
 *		ISC_EVENT_COMMON(struct isc_event);
 *		...
 *	};
 * \endcode
 *
 * Requires:
 *\li	'size' >= sizeof(struct isc_event)
 *\li	'action' to be non NULL
 *
 * Returns:
 *\li	a pointer to a initialized structure of the requested size.
 *\li	NULL if unable to allocate memory.
 */

void
isc_event_free(isc_event_t **);

ISC_LANG_ENDDECLS

#endif /* ISC_EVENT_H */
