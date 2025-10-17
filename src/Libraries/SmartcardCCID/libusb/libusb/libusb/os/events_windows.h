/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 12, 2023.
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
#ifndef LIBUSB_EVENTS_WINDOWS_H
#define LIBUSB_EVENTS_WINDOWS_H

typedef HANDLE usbi_os_handle_t;
#define USBI_OS_HANDLE_FORMAT_STRING	"HANDLE %p"

typedef struct usbi_event {
	HANDLE hEvent;
} usbi_event_t;
#define USBI_EVENT_OS_HANDLE(e)	((e)->hEvent)
#define USBI_EVENT_POLL_EVENTS	0
#define USBI_INVALID_EVENT	{ INVALID_HANDLE_VALUE }

#define HAVE_OS_TIMER 1
typedef struct usbi_timer {
	HANDLE hTimer;
} usbi_timer_t;
#define USBI_TIMER_OS_HANDLE(t)	((t)->hTimer)
#define USBI_TIMER_POLL_EVENTS	0

static inline int usbi_timer_valid(usbi_timer_t *timer)
{
	return timer->hTimer != NULL;
}

#endif
