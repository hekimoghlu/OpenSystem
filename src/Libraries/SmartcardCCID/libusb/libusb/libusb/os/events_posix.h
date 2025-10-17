/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 10, 2024.
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
#ifndef LIBUSB_EVENTS_POSIX_H
#define LIBUSB_EVENTS_POSIX_H

#include <poll.h>

typedef int usbi_os_handle_t;
#define USBI_OS_HANDLE_FORMAT_STRING	"fd %d"

#ifdef HAVE_EVENTFD
typedef struct usbi_event {
	int eventfd;
} usbi_event_t;
#define USBI_EVENT_OS_HANDLE(e)	((e)->eventfd)
#define USBI_EVENT_POLL_EVENTS	POLLIN
#define USBI_INVALID_EVENT	{ -1 }
#else
typedef struct usbi_event {
	int pipefd[2];
} usbi_event_t;
#define USBI_EVENT_OS_HANDLE(e)	((e)->pipefd[0])
#define USBI_EVENT_POLL_EVENTS	POLLIN
#define USBI_INVALID_EVENT	{ { -1, -1 } }
#endif

#ifdef HAVE_TIMERFD
#define HAVE_OS_TIMER 1
typedef struct usbi_timer {
	int timerfd;
} usbi_timer_t;
#define USBI_TIMER_OS_HANDLE(t)	((t)->timerfd)
#define USBI_TIMER_POLL_EVENTS	POLLIN

static inline int usbi_timer_valid(usbi_timer_t *timer)
{
	return timer->timerfd >= 0;
}
#endif

#endif
