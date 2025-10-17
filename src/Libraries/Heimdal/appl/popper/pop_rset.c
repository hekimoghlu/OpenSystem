/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 3, 2022.
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
 *  rset:   Unflag all messages flagged for deletion in a POP maildrop
 */

int
pop_rset (POP *p)
{
    MsgInfoList     *   mp;         /*  Pointer to the message info list */
    int		        i;

    /*  Unmark all the messages */
    for (i = p->msg_count, mp = p->mlp; i > 0; i--, mp++)
        mp->flags &= ~DEL_FLAG;

    /*  Reset the messages-deleted and bytes-deleted counters */
    p->msgs_deleted = 0;
    p->bytes_deleted = 0;

    /*  Reset the last-message-access flag */
    p->last_msg = 0;

    return (pop_msg(p,POP_SUCCESS,"Maildrop has %u messages (%ld octets)",
		    p->msg_count, p->drop_size));
}
