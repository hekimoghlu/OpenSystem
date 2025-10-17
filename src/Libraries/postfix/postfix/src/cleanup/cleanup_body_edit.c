/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 19, 2023.
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

/* Utility library. */

#include <msg.h>
#include <mymalloc.h>
#include <vstream.h>
#include <vstring.h>

/* Global library. */

#include <rec_type.h>
#include <record.h>

/* Application-specific. */

#include <cleanup.h>

#define LEN(s) VSTRING_LEN(s)

static int cleanup_body_edit_ptr_rec_len;

/* cleanup_body_edit_start - rewrite body region pool */

int     cleanup_body_edit_start(CLEANUP_STATE *state)
{
    const char *myname = "cleanup_body_edit_start";
    CLEANUP_REGION *curr_rp;

    /*
     * Calculate the payload size sans body.
     */
    state->cont_length = state->body_offset - state->data_offset;

    /*
     * One-time initialization.
     */
    if (state->body_regions == 0) {
	REC_SPACE_NEED(REC_TYPE_PTR_PAYL_SIZE, cleanup_body_edit_ptr_rec_len);
	cleanup_region_init(state);
    }

    /*
     * Return all body regions to the free pool.
     */
    cleanup_region_return(state, state->body_regions);

    /*
     * Select the first region. XXX This will usually be the original body
     * segment, but we must not count on that. Region assignments may change
     * when header editing also uses queue file regions. XXX We don't really
     * know if the first region will be large enough to hold the first body
     * text record, but this problem is so rare that we will not complicate
     * the code for it. If the first region is too small then we will simply
     * waste it.
     */
    curr_rp = state->curr_body_region = state->body_regions =
	cleanup_region_open(state, cleanup_body_edit_ptr_rec_len);

    /*
     * Link the first body region to the last message header.
     */
    if (vstream_fseek(state->dst, state->append_hdr_pt_offset, SEEK_SET) < 0) {
	msg_warn("%s: seek file %s: %m", myname, cleanup_path);
	return (-1);
    }
    state->append_hdr_pt_target = curr_rp->start;
    rec_fprintf(state->dst, REC_TYPE_PTR, REC_TYPE_PTR_FORMAT,
		(long) state->append_hdr_pt_target);

    /*
     * Move the file write pointer to the start of the current region.
     */
    if (vstream_ftell(state->dst) != curr_rp->start
	&& vstream_fseek(state->dst, curr_rp->start, SEEK_SET) < 0) {
	msg_warn("%s: seek file %s: %m", myname, cleanup_path);
	return (-1);
    }
    return (0);
}

/* cleanup_body_edit_write - add record to body region pool */

int     cleanup_body_edit_write(CLEANUP_STATE *state, int rec_type,
				        VSTRING *buf)
{
    const char *myname = "cleanup_body_edit_write";
    CLEANUP_REGION *curr_rp = state->curr_body_region;
    CLEANUP_REGION *next_rp;
    off_t   space_used;
    ssize_t space_needed;
    ssize_t rec_len;

    if (msg_verbose)
	msg_info("%s: where %ld, buflen %ld region start %ld len %ld",
		 myname, (long) curr_rp->write_offs, (long) LEN(buf),
		 (long) curr_rp->start, (long) curr_rp->len);

    /*
     * Switch to the next body region if we filled up the current one (we
     * always append to an open-ended region). Besides space to write the
     * specified record, we need to leave space for a final pointer record
     * that will link this body region to the next region or to the content
     * terminator record.
     */
    if (curr_rp->len > 0) {
	space_used = curr_rp->write_offs - curr_rp->start;
	REC_SPACE_NEED(LEN(buf), rec_len);
	space_needed = rec_len + cleanup_body_edit_ptr_rec_len;
	if (space_needed > curr_rp->len - space_used) {

	    /*
	     * Update the payload size. Connect the filled up body region to
	     * its successor.
	     */
	    state->cont_length += space_used;
	    next_rp = cleanup_region_open(state, space_needed);
	    if (msg_verbose)
		msg_info("%s: link %ld -> %ld", myname,
			 (long) curr_rp->write_offs, (long) next_rp->start);
	    rec_fprintf(state->dst, REC_TYPE_PTR, REC_TYPE_PTR_FORMAT,
			(long) next_rp->start);
	    curr_rp->write_offs = vstream_ftell(state->dst);
	    cleanup_region_close(state, curr_rp);
	    curr_rp->next = next_rp;

	    /*
	     * Select the new body region.
	     */
	    state->curr_body_region = curr_rp = next_rp;
	    if (vstream_fseek(state->dst, curr_rp->start, SEEK_SET) < 0) {
		msg_warn("%s: seek file %s: %m", myname, cleanup_path);
		return (-1);
	    }
	}
    }

    /*
     * Finally, output the queue file record.
     */
    CLEANUP_OUT_BUF(state, REC_TYPE_NORM, buf);
    curr_rp->write_offs = vstream_ftell(state->dst);

    return (0);
}

/* cleanup_body_edit_finish - wrap up body region pool */

int     cleanup_body_edit_finish(CLEANUP_STATE *state)
{
    CLEANUP_REGION *curr_rp = state->curr_body_region;

    /*
     * Update the payload size.
     */
    state->cont_length += curr_rp->write_offs - curr_rp->start;

    /*
     * Link the last body region to the content terminator record.
     */
    rec_fprintf(state->dst, REC_TYPE_PTR, REC_TYPE_PTR_FORMAT,
		(long) state->xtra_offset);
    curr_rp->write_offs = vstream_ftell(state->dst);
    cleanup_region_close(state, curr_rp);

    return (CLEANUP_OUT_OK(state) ? 0 : -1);
}
