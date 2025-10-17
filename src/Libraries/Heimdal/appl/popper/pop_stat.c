/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 19, 2024.
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
#include <popper.h>
RCSID("$Id$");

/*
 *  stat:   Display the status of a POP maildrop to its client
 */

int
pop_stat (POP *p)
{
#ifdef DEBUG
    if (p->debug) pop_log(p,POP_DEBUG,"%d message(s) (%ld octets).",
			  p->msg_count-p->msgs_deleted,
			  p->drop_size-p->bytes_deleted);
#endif /* DEBUG */
    return (pop_msg (p,POP_SUCCESS,
		     "%d %ld",
		     p->msg_count-p->msgs_deleted,
		     p->drop_size-p->bytes_deleted));
}
