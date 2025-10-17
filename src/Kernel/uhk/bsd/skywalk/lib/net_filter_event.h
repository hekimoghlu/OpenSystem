/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 3, 2021.
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
#include <stdbool.h>

#include <sys/types.h>
#include <sys/eventhandler.h>

enum net_filter_event_subsystems : uint32_t {
	NET_FILTER_EVENT_PF = (1 << 0),
	NET_FILTER_EVENT_SOCKET = (1 << 1),
	NET_FILTER_EVENT_INTERFACE = (1 << 2),
	NET_FILTER_EVENT_IP = (1 << 3),
	NET_FILTER_EVENT_ALF = (1 << 4),
	NET_FILTER_EVENT_PARENTAL_CONTROLS = (1 << 5),
	NET_FILTER_EVENT_PF_PRIVATE_PROXY = (1 << 6),
};

/* Marks subsystem filtering state. */
void
net_filter_event_mark(enum net_filter_event_subsystems subsystem, bool compatible);

typedef void (*net_filter_event_callback_t) (struct eventhandler_entry_arg,
    enum net_filter_event_subsystems);

/* Registers a function to be called when state changes. */
void
net_filter_event_register(net_filter_event_callback_t callback);

/* Gets the state of the filters. */
enum net_filter_event_subsystems
net_filter_event_get_state(void);
