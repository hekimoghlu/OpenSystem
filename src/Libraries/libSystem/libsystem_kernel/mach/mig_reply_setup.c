/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 28, 2025.
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
 * Mach Operating System
 * Copyright (c) 1991 Carnegie Mellon University
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
 * Routine to set up a MiG reply message from a request message.
 *
 * Knows about the MiG reply message ID convention:
 *	reply_id = request_id + 100
 *
 * For typed IPC sets up the RetCode type field.  Does NOT set a
 * return code value.
 */

#include <mach/mach.h>
#include <mach/message.h>
#include <mach/mig_errors.h>

void
mig_reply_setup(mach_msg_header_t *request, mach_msg_header_t *reply)
{
#define InP     (request)
#define OutP    ((mig_reply_error_t *) reply)

	OutP->Head.msgh_bits =
	    MACH_MSGH_BITS(MACH_MSGH_BITS_LOCAL(InP->msgh_bits), 0);
	OutP->Head.msgh_size = sizeof(mig_reply_error_t);
	OutP->Head.msgh_remote_port = InP->msgh_local_port;
	OutP->Head.msgh_local_port  = MACH_PORT_NULL;
	OutP->Head.msgh_id = InP->msgh_id + 100;
	OutP->NDR = NDR_record;
}
