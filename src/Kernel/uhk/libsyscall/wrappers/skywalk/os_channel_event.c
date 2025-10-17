/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 17, 2023.
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
#include <stddef.h>
#include <stdint.h>
#include <errno.h>
#include <skywalk/os_packet.h>
#include <skywalk/os_channel_event.h>

#ifndef LIBSYSCALL_INTERFACE
#error "LIBSYSCALL_INTERFACE not defined"
#endif /* !LIBSYSCALL_INTERFACE */

int
os_channel_event_get_next_event(const os_channel_event_handle_t event_handle,
    const os_channel_event_t prev_event, os_channel_event_t *event)
{
	struct __kern_channel_event *cev, *pev;
	buflet_t buflet;
	uint16_t bdlen;
	char *baddr, *estart;

	*event = NULL;
	if (!event_handle) {
		return EINVAL;
	}
	buflet = os_packet_get_next_buflet(event_handle, NULL);
	if (__improbable(buflet == NULL)) {
		return EINVAL;
	}
	baddr = os_buflet_get_object_address(buflet);
	if (__improbable(baddr == NULL)) {
		return ENXIO;
	}
	bdlen = os_buflet_get_data_length(buflet);
	baddr += os_buflet_get_data_offset(buflet);
	estart = baddr + __KERN_CHANNEL_EVENT_OFFSET;
	pev = (struct __kern_channel_event *)prev_event;
	if (pev == NULL) {
		cev = (struct __kern_channel_event *)estart;
	} else {
		if ((pev->ev_flags & CHANNEL_EVENT_FLAG_MORE_EVENT) == 0) {
			return ENODATA;
		}
		cev = (struct __kern_channel_event *)((char *)pev + sizeof(*pev) +
		    pev->ev_dlen);
	}
	if (__improbable((char *)cev < estart)) {
		return ENXIO;
	}
	if (__improbable((cev->ev_dlen + (char *)cev) > (baddr + bdlen))) {
		return ENXIO;
	}
	*event = (os_channel_event_t)cev;
	return 0;
}

int
os_channel_event_get_event_data(const os_channel_event_t event,
    struct os_channel_event_data *event_data)
{
	struct __kern_channel_event *kev;

	if (__improbable(event == 0 || event_data == NULL)) {
		return EINVAL;
	}
	kev = (struct __kern_channel_event *)event;
	if (__improbable(kev->ev_type < CHANNEL_EVENT_MIN ||
	    kev->ev_type > CHANNEL_EVENT_MAX)) {
		return ENXIO;
	}
	event_data->event_type = kev->ev_type;
	event_data->event_more =
	    (kev->ev_flags & CHANNEL_EVENT_FLAG_MORE_EVENT) != 0;
	event_data->event_data_length = kev->ev_dlen;
	event_data->event_data = kev->ev_data;
	return 0;
}
