/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 27, 2024.
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
 * @OSF_COPYRIGHT@
 */
/*
 * Mach Operating System
 * Copyright (c) 1991,1990,1989,1988,1987 Carnegie Mellon University
 * All Rights Reserved.
 *
 * Permission to use, copy, modify and distribute this software and its
 * documentation is hereby granted, provided that both the copyright
 * notice and this permission notice appear in all copies of the
 * software, derivative works or modified versions, and any portions
 * thereof, and that both notices appear in supporting documentation.
 *
 * CARNEGIE MELLON ALLOWS FREE USE OF THIS SOFTWARE IN ITS "AS IS"
 * CONDITION.  CARNEGIE MELLON DISCLAIMS ANY LIABILITY OF ANY KIND FOR
 * ANY DAMAGES WHATSOEVER RESULTING FROM THE USE OF THIS SOFTWARE.
 *
 * Carnegie Mellon requests users of this software to return to
 *
 *  Software Distribution Coordinator  or  Software.Distribution@CS.CMU.EDU
 *  School of Computer Science
 *  Carnegie Mellon University
 *  Pittsburgh PA 15213-3890
 *
 * any improvements or extensions that they make and grant Carnegie Mellon
 * the rights to redistribute these changes.
 */
/*
 */
/*
 *	File:	mach/notify.h
 *
 *	Kernel notification message definitions.
 */

#ifndef _MACH_NOTIFY_H_
#define _MACH_NOTIFY_H_

#include <mach/port.h>
#include <mach/message.h>
#include <mach/ndr.h>

/*
 *  An alternative specification of the notification interface
 *  may be found in mach/notify.defs.
 */

#define MACH_NOTIFY_FIRST               0100
#define MACH_NOTIFY_PORT_DELETED        (MACH_NOTIFY_FIRST + 001)
/* A send or send-once right was deleted. */
#define MACH_NOTIFY_SEND_POSSIBLE       (MACH_NOTIFY_FIRST + 002)
/* Now possible to send using specified right */
#define MACH_NOTIFY_PORT_DESTROYED      (MACH_NOTIFY_FIRST + 005)
/* A receive right was (would have been) deallocated */
#define MACH_NOTIFY_NO_SENDERS          (MACH_NOTIFY_FIRST + 006)
/* Receive right has no extant send rights */
#define MACH_NOTIFY_SEND_ONCE           (MACH_NOTIFY_FIRST + 007)
/* An extant send-once right died */
#define MACH_NOTIFY_DEAD_NAME           (MACH_NOTIFY_FIRST + 010)
/* Send or send-once right died, leaving a dead-name */
#define MACH_NOTIFY_LAST                (MACH_NOTIFY_FIRST + 015)

typedef mach_port_t notify_port_t;

/*
 * Hard-coded message structures for receiving Mach port notification
 * messages.  However, they are not actual large enough to receive
 * the largest trailers current exported by Mach IPC (so they cannot
 * be used for space allocations in situations using these new larger
 * trailers).  Instead, the MIG-generated server routines (and
 * related prototypes should be used).
 */
typedef struct {
	mach_msg_header_t   not_header;
	NDR_record_t        NDR;
	mach_port_name_t not_port;/* MACH_MSG_TYPE_PORT_NAME */
	mach_msg_format_0_trailer_t trailer;
} mach_port_deleted_notification_t;

typedef struct {
	mach_msg_header_t   not_header;
	NDR_record_t        NDR;
	mach_port_name_t not_port;/* MACH_MSG_TYPE_PORT_NAME */
	mach_msg_format_0_trailer_t trailer;
} mach_send_possible_notification_t;

typedef struct {
	mach_msg_header_t   not_header;
	mach_msg_body_t     not_body;
	mach_msg_port_descriptor_t not_port;/* MACH_MSG_TYPE_PORT_RECEIVE */
	mach_msg_format_0_trailer_t trailer;
} mach_port_destroyed_notification_t;

typedef struct {
	mach_msg_header_t   not_header;
	NDR_record_t        NDR;
	mach_msg_type_number_t not_count;
	mach_msg_format_0_trailer_t trailer;
} mach_no_senders_notification_t;

typedef struct {
	mach_msg_header_t   not_header;
	mach_msg_format_0_trailer_t trailer;
} mach_send_once_notification_t;

typedef struct {
	mach_msg_header_t   not_header;
	NDR_record_t        NDR;
	mach_port_name_t not_port;/* MACH_MSG_TYPE_PORT_NAME */
	mach_msg_format_0_trailer_t trailer;
} mach_dead_name_notification_t;

#endif  /* _MACH_NOTIFY_H_ */
