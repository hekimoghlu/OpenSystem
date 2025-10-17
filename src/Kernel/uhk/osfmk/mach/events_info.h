/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 17, 2024.
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
 * @OSF_FREE_COPYRIGHT@
 */
/*
 *	Machine-independent event information structures and definitions.
 *
 *	The definitions in this file are exported to the user.  The kernel
 *	will translate its internal data structures to these structures
 *	as appropriate.
 *
 *	This data structure is used to track events that occur during
 *	thread execution, and to summarize this information for tasks.
 */

#ifndef _MACH_EVENTS_INFO_H_
#define _MACH_EVENTS_INFO_H_

#include <mach/message.h>

struct events_info {
	integer_t       faults;         /* number of page faults */
	integer_t       zero_fills;     /* number of zero fill pages */
	integer_t       reactivations;  /* number of reactivated pages */
	integer_t       pageins;        /* number of actual pageins */
	integer_t       cow_faults;     /* number of copy-on-write faults */
	integer_t       messages_sent;  /* number of messages sent */
	integer_t       messages_received; /* number of messages received */
};
typedef struct events_info              events_info_data_t;
typedef struct events_info              *events_info_t;
#define EVENTS_INFO_COUNT       ((mach_msg_type_number_t) \
	        (sizeof(events_info_data_t) / sizeof(integer_t)))

#endif  /*_MACH_EVENTS_INFO_H_*/
