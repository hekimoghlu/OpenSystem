/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 19, 2025.
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

#ifdef UIDL
/*
 *  uidl:   Uidl the contents of a POP maildrop
 */

int
pop_uidl (POP *p)
{
    MsgInfoList         *   mp;         /*  Pointer to message info list */
    int		            i;
    int			    msg_num;

    /*  Was a message number provided? */
    if (p->parm_count > 0) {
        msg_num = atoi(p->pop_parm[1]);

        /*  Is requested message out of range? */
        if ((msg_num < 1) || (msg_num > p->msg_count))
            return (pop_msg (p,POP_FAILURE,
                "Message %d does not exist.",msg_num));

        /*  Get a pointer to the message in the message list */
        mp = &p->mlp[msg_num-1];

        /*  Is the message already flagged for deletion? */
        if (mp->flags & DEL_FLAG)
            return (pop_msg (p,POP_FAILURE,
                "Message %d has been deleted.",msg_num));

        /*  Display message information */
        return (pop_msg(p,POP_SUCCESS,"%u %s",msg_num,mp->msg_id));
    }

    /*  Display the entire list of messages */
    pop_msg(p,POP_SUCCESS,
	    "%d messages (%ld octets)",
            p->msg_count-p->msgs_deleted,
	    p->drop_size-p->bytes_deleted);

    /*  Loop through the message information list.  Skip deleted messages */
    for (i = p->msg_count, mp = p->mlp; i > 0; i--, mp++) {
        if (!(mp->flags & DEL_FLAG))
            fprintf(p->output,"%u %s\r\n",mp->number,mp->msg_id);
    }

    /*  "." signals the end of a multi-line transmission */
    fprintf(p->output,".\r\n");
    fflush(p->output);

    return(POP_SUCCESS);
}
#endif /* UIDL */
