/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 22, 2024.
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
/* System library. */

#include <sys_defs.h>
#include <errno.h>

/* Utility library. */

#include <msg.h>

/* Global library. */

#include <cleanup_user.h>
#include <rec_type.h>

/* Application-specific. */

#include "cleanup.h"

/* cleanup_final - final queue file content updates */

void    cleanup_final(CLEANUP_STATE *state)
{
    const char *myname = "cleanup_final";

    /*
     * vstream_fseek() would flush the buffer anyway, but the code just reads
     * better if we flush first, because it makes seek error handling more
     * straightforward.
     */
    if (vstream_fflush(state->dst)) {
	if (errno == EFBIG) {
	    msg_warn("%s: queue file size limit exceeded", state->queue_id);
	    state->errs |= CLEANUP_STAT_SIZE;
	} else {
	    msg_warn("%s: write queue file: %m", state->queue_id);
	    state->errs |= CLEANUP_STAT_WRITE;
	}
	return;
    }

    /*
     * Update the preliminary message size and count fields with the actual
     * values.
     */
    if (vstream_fseek(state->dst, 0L, SEEK_SET) < 0)
	msg_fatal("%s: vstream_fseek %s: %m", myname, cleanup_path);
    cleanup_out_format(state, REC_TYPE_SIZE, REC_TYPE_SIZE_FORMAT,
	    (REC_TYPE_SIZE_CAST1) (state->xtra_offset - state->data_offset),
		       (REC_TYPE_SIZE_CAST2) state->data_offset,
		       (REC_TYPE_SIZE_CAST3) state->rcpt_count,
		       (REC_TYPE_SIZE_CAST4) state->qmgr_opts,
		       (REC_TYPE_SIZE_CAST5) state->cont_length,
		       (REC_TYPE_SIZE_CAST6) state->smtputf8);
}
