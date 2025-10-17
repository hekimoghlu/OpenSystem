/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 3, 2022.
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

int
pop_xover (POP *p)
{
#ifdef XOVER
    MsgInfoList         *   mp;         /*  Pointer to message info list */
    int		            i;

    pop_msg(p,POP_SUCCESS,
	    "%d messages (%ld octets)",
            p->msg_count-p->msgs_deleted,
	    p->drop_size-p->bytes_deleted);

    /*  Loop through the message information list.  Skip deleted messages */
    for (i = p->msg_count, mp = p->mlp; i > 0; i--, mp++) {
        if (!(mp->flags & DEL_FLAG))
            fprintf(p->output,"%u\t%s\t%s\t%s\t%s\t%lu\t%u\r\n",
		    mp->number,
		    mp->subject,
		    mp->from,
		    mp->date,
		    mp->msg_id,
		    mp->length,
		    mp->lines);
    }

    /*  "." signals the end of a multi-line transmission */
    fprintf(p->output,".\r\n");
    fflush(p->output);

    return(POP_SUCCESS);
#else
    return pop_msg(p, POP_FAILURE, "Command not implemented.");
#endif
}
