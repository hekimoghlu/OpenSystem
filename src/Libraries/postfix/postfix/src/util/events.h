/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 23, 2024.
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

#ifndef _EVENTS_H_INCLUDED_
#define _EVENTS_H_INCLUDED_

/*++
/* NAME
/*	events 3h
/* SUMMARY
/*	event manager
/* SYNOPSIS
/*	#include <events.h>
/* DESCRIPTION
/* .nf

 /*
  * System library.
  */
#include <time.h>

 /*
  * External interface.
  */
typedef void (*EVENT_NOTIFY_FN) (int, void *);

#define EVENT_NOTIFY_TIME_FN EVENT_NOTIFY_FN	/* legacy */
#define EVENT_NOTIFY_RDWR_FN EVENT_NOTIFY_FN	/* legacy */

extern time_t event_time(void);
extern void event_enable_read(int, EVENT_NOTIFY_RDWR_FN, void *);
extern void event_enable_write(int, EVENT_NOTIFY_RDWR_FN, void *);
extern void event_disable_readwrite(int);
extern time_t event_request_timer(EVENT_NOTIFY_TIME_FN, void *, int);
extern int event_cancel_timer(EVENT_NOTIFY_TIME_FN, void *);
extern void event_loop(int);
extern void event_drain(int);
extern void event_fork(void);

 /*
  * Event codes.
  */
#define EVENT_READ	(1<<0)		/* read event */
#define EVENT_WRITE	(1<<1)		/* write event */
#define EVENT_XCPT	(1<<2)		/* exception */
#define EVENT_TIME	(1<<3)		/* timer event */

#define EVENT_ERROR	EVENT_XCPT

 /*
  * Dummies.
  */
#define EVENT_NULL_TYPE		(0)
#define EVENT_NULL_CONTEXT	((void *) 0)
#define EVENT_NULL_DELAY	(0)

/* LICENSE
/* .ad
/* .fi
/*	The Secure Mailer license must be distributed with this software.
/* AUTHOR(S)
/*	Wietse Venema
/*	IBM T.J. Watson Research
/*	P.O. Box 704
/*	Yorktown Heights, NY 10598, USA
/* CREATION DATE
/*	Wed Jan 29 17:00:03 EST 1997
/*--*/

#endif
