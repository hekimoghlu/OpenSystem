/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 17, 2025.
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
 *  dele:   Delete a message from the POP maildrop
 */
int
pop_dele (POP *p)
{
    MsgInfoList     *   mp;         /*  Pointer to message info list */
    int                 msg_num;

    /*  Convert the message number parameter to an integer */
    msg_num = atoi(p->pop_parm[1]);

    /*  Is requested message out of range? */
    if ((msg_num < 1) || (msg_num > p->msg_count))
        return (pop_msg (p,POP_FAILURE,"Message %d does not exist.",msg_num));

    /*  Get a pointer to the message in the message list */
    mp = &(p->mlp[msg_num-1]);

    /*  Is the message already flagged for deletion? */
    if (mp->flags & DEL_FLAG)
        return (pop_msg (p,POP_FAILURE,"Message %d has already been deleted.",
            msg_num));

    /*  Flag the message for deletion */
    mp->flags |= DEL_FLAG;

#ifdef DEBUG
    if(p->debug)
        pop_log(p, POP_DEBUG,
		"Deleting message %u at offset %ld of length %ld\n",
		mp->number, mp->offset, mp->length);
#endif /* DEBUG */

    /*  Update the messages_deleted and bytes_deleted counters */
    p->msgs_deleted++;
    p->bytes_deleted += mp->length;

    /*  Update the last-message-accessed number if it is lower than
        the deleted message */
    if (p->last_msg < msg_num) p->last_msg = msg_num;

    return (pop_msg (p,POP_SUCCESS,"Message %d has been deleted.",msg_num));
}

#ifdef XDELE
/* delete a range of messages */
int
pop_xdele(POP *p)
{
    MsgInfoList     *   mp;         /*  Pointer to message info list */

    int msg_min, msg_max;
    int i;


    msg_min = atoi(p->pop_parm[1]);
    if(p->parm_count == 1)
	msg_max = msg_min;
    else
	msg_max = atoi(p->pop_parm[2]);

    if (msg_min < 1)
        return (pop_msg (p,POP_FAILURE,"Message %d does not exist.",msg_min));
    if(msg_max > p->msg_count)
        return (pop_msg (p,POP_FAILURE,"Message %d does not exist.",msg_max));
    for(i = msg_min; i <= msg_max; i++) {

	/*  Get a pointer to the message in the message list */
	mp = &(p->mlp[i - 1]);

	/*  Is the message already flagged for deletion? */
	if (mp->flags & DEL_FLAG)
	    continue; /* no point in returning error */
	/*  Flag the message for deletion */
	mp->flags |= DEL_FLAG;

#ifdef DEBUG
	if(p->debug)
	    pop_log(p, POP_DEBUG,
		    "Deleting message %u at offset %ld of length %ld\n",
		    mp->number, mp->offset, mp->length);
#endif /* DEBUG */

	/*  Update the messages_deleted and bytes_deleted counters */
	p->msgs_deleted++;
	p->bytes_deleted += mp->length;
    }

    /*  Update the last-message-accessed number if it is lower than
	the deleted message */
    if (p->last_msg < msg_max) p->last_msg = msg_max;

    return (pop_msg (p,POP_SUCCESS,"Messages %d-%d has been deleted.",
		     msg_min, msg_max));

}
#endif /* XDELE */
