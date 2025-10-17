/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 26, 2024.
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
#include <sys/stat.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

/* Utility library. */

#include <msg.h>
#include <vstring.h>
#include <vstream.h>
#include <mymalloc.h>
#include <stringops.h>
#include <set_eugid.h>

/* Global library. */

#include <mail_copy.h>
#include <mbox_open.h>
#include <defer.h>
#include <sent.h>
#include <mail_params.h>
#include <mail_addr_find.h>
#include <dsn_util.h>

/* Application-specific. */

#include "virtual.h"

#define YES	1
#define NO	0

/* deliver_mailbox_file - deliver to recipient mailbox */

static int deliver_mailbox_file(LOCAL_STATE state, USER_ATTR usr_attr)
{
    const char *myname = "deliver_mailbox_file";
    DSN_BUF *why = state.msg_attr.why;
    MBOX   *mp;
    int     mail_copy_status;
    int     deliver_status;
    int     copy_flags;
    struct stat st;

    /*
     * Make verbose logging easier to understand.
     */
    state.level++;
    if (msg_verbose)
	MSG_LOG_STATE(myname, state);

    /*
     * Don't deliver trace-only requests.
     */
    if (DEL_REQ_TRACE_ONLY(state.request->flags)) {
	dsb_simple(why, "2.0.0", "delivers to mailbox");
	return (sent(BOUNCE_FLAGS(state.request),
		     SENT_ATTR(state.msg_attr)));
    }

    /*
     * Initialize. Assume the operation will fail. Set the delivered
     * attribute to reflect the final recipient.
     */
    if (vstream_fseek(state.msg_attr.fp, state.msg_attr.offset, SEEK_SET) < 0)
	msg_fatal("seek message file %s: %m", VSTREAM_PATH(state.msg_attr.fp));
    state.msg_attr.delivered = state.msg_attr.rcpt.address;
    mail_copy_status = MAIL_COPY_STAT_WRITE;

    /*
     * Lock the mailbox and open/create the mailbox file.
     * 
     * Write the file as the recipient, so that file quota work.
     */
    copy_flags = MAIL_COPY_MBOX;

    set_eugid(usr_attr.uid, usr_attr.gid);
    mp = mbox_open(usr_attr.mailbox, O_APPEND | O_WRONLY | O_CREAT,
		   S_IRUSR | S_IWUSR, &st, -1, -1,
		   virtual_mbox_lock_mask, "4.2.0", why);
    if (mp != 0) {
	if (S_ISREG(st.st_mode) == 0) {
	    vstream_fclose(mp->fp);
	    msg_warn("recipient %s: destination %s is not a regular file",
		     state.msg_attr.rcpt.address, usr_attr.mailbox);
	    dsb_simple(why, "5.3.5", "mail system configuration error");
	} else if (var_strict_mbox_owner && st.st_uid != usr_attr.uid) {
	    vstream_fclose(mp->fp);
	    dsb_simple(why, "4.2.0",
	      "destination %s is not owned by recipient", usr_attr.mailbox);
	    msg_warn("specify \"%s = no\" to ignore mailbox ownership mismatch",
		     VAR_STRICT_MBOX_OWNER);
	} else {
	    if (vstream_fseek(mp->fp, (off_t) 0, SEEK_END) < 0)
		msg_fatal("%s: seek queue file %s: %m",
			  myname, VSTREAM_PATH(mp->fp));
	    mail_copy_status = mail_copy(COPY_ATTR(state.msg_attr), mp->fp,
					 copy_flags, "\n", why);
	}
	mbox_release(mp);
    }
    set_eugid(var_owner_uid, var_owner_gid);

    /*
     * As the mail system, bounce, defer delivery, or report success.
     */
    if (mail_copy_status & MAIL_COPY_STAT_CORRUPT) {
	deliver_status = DEL_STAT_DEFER;
    } else if (mail_copy_status != 0) {
	vstring_sprintf_prepend(why->reason, "delivery failed to mailbox %s: ",
				usr_attr.mailbox);
	deliver_status =
	    (STR(why->status)[0] == '4' ?
	     defer_append : bounce_append)
	    (BOUNCE_FLAGS(state.request),
	     BOUNCE_ATTR(state.msg_attr));
    } else {
	dsb_simple(why, "2.0.0", "delivered to mailbox");
	deliver_status = sent(BOUNCE_FLAGS(state.request),
			      SENT_ATTR(state.msg_attr));
    }
    return (deliver_status);
}

/* deliver_mailbox - deliver to recipient mailbox */

int     deliver_mailbox(LOCAL_STATE state, USER_ATTR usr_attr, int *statusp)
{
    const char *myname = "deliver_mailbox";
    const char *mailbox_res;
    const char *uid_res;
    const char *gid_res;
    DSN_BUF *why = state.msg_attr.why;
    long    n;

    /*
     * Make verbose logging easier to understand.
     */
    state.level++;
    if (msg_verbose)
	MSG_LOG_STATE(myname, state);

    /*
     * Sanity check.
     */
    if (*var_virt_mailbox_base != '/')
	msg_fatal("do not specify relative pathname: %s = %s",
		  VAR_VIRT_MAILBOX_BASE, var_virt_mailbox_base);

    /*
     * Look up the mailbox location. Bounce if not found, defer in case of
     * trouble.
     */
#define IGNORE_EXTENSION ((char **) 0)

    mailbox_res = mail_addr_find(virtual_mailbox_maps, state.msg_attr.user,
				 IGNORE_EXTENSION);
    if (mailbox_res == 0) {
	if (virtual_mailbox_maps->error == 0)
	    return (NO);
	msg_warn("table %s: lookup %s: %m", virtual_mailbox_maps->title,
		 state.msg_attr.user);
	dsb_simple(why, "4.3.5", "mail system configuration error");
	*statusp = defer_append(BOUNCE_FLAGS(state.request),
				BOUNCE_ATTR(state.msg_attr));
	return (YES);
    }
    usr_attr.mailbox = concatenate(var_virt_mailbox_base, "/",
				   mailbox_res, (char *) 0);

#define RETURN(res) { myfree(usr_attr.mailbox); return (res); }

    /*
     * Look up the mailbox owner rights. Defer in case of trouble.
     */
    uid_res = mail_addr_find(virtual_uid_maps, state.msg_attr.user,
			     IGNORE_EXTENSION);
    if (uid_res == 0) {
	msg_warn("recipient %s: not found in %s",
		 state.msg_attr.user, virtual_uid_maps->title);
	dsb_simple(why, "4.3.5", "mail system configuration error");
	*statusp = defer_append(BOUNCE_FLAGS(state.request),
				BOUNCE_ATTR(state.msg_attr));
	RETURN(YES);
    }
    if ((n = atol(uid_res)) < var_virt_minimum_uid) {
	msg_warn("recipient %s: bad uid %s in %s",
		 state.msg_attr.user, uid_res, virtual_uid_maps->title);
	dsb_simple(why, "4.3.5", "mail system configuration error");
	*statusp = defer_append(BOUNCE_FLAGS(state.request),
				BOUNCE_ATTR(state.msg_attr));
	RETURN(YES);
    }
    usr_attr.uid = (uid_t) n;

    /*
     * Look up the mailbox group rights. Defer in case of trouble.
     */
    gid_res = mail_addr_find(virtual_gid_maps, state.msg_attr.user,
			     IGNORE_EXTENSION);
    if (gid_res == 0) {
	msg_warn("recipient %s: not found in %s",
		 state.msg_attr.user, virtual_gid_maps->title);
	dsb_simple(why, "4.3.5", "mail system configuration error");
	*statusp = defer_append(BOUNCE_FLAGS(state.request),
				BOUNCE_ATTR(state.msg_attr));
	RETURN(YES);
    }
    if ((n = atol(gid_res)) <= 0) {
	msg_warn("recipient %s: bad gid %s in %s",
		 state.msg_attr.user, gid_res, virtual_gid_maps->title);
	dsb_simple(why, "4.3.5", "mail system configuration error");
	*statusp = defer_append(BOUNCE_FLAGS(state.request),
				BOUNCE_ATTR(state.msg_attr));
	RETURN(YES);
    }
    usr_attr.gid = (gid_t) n;

    if (msg_verbose)
	msg_info("%s[%d]: set user_attr: %s, uid = %u, gid = %u",
		 myname, state.level, usr_attr.mailbox,
		 (unsigned) usr_attr.uid, (unsigned) usr_attr.gid);

    /*
     * Deliver to mailbox or to maildir.
     */
#define LAST_CHAR(s) (s[strlen(s) - 1])

    if (LAST_CHAR(usr_attr.mailbox) == '/')
	*statusp = deliver_maildir(state, usr_attr);
    else
	*statusp = deliver_mailbox_file(state, usr_attr);

    /*
     * Cleanup.
     */
    RETURN(YES);
}
