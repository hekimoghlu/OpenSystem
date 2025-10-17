/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 11, 2023.
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
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>

/* Utility library. */

#include <msg.h>
#include <htable.h>
#include <mymalloc.h>
#include <vstream.h>
#include <open_as.h>
#include <stat_as.h>
#include <iostuff.h>
#include <mypwd.h>

/* Global library. */

#include <bounce.h>
#include <defer.h>
#include <been_here.h>
#include <mail_params.h>
#include <ext_prop.h>
#include <sent.h>

/* Application-specific. */

#include "local.h"

/* deliver_include - open include file and deliver */

int     deliver_include(LOCAL_STATE state, USER_ATTR usr_attr, char *path)
{
    const char *myname = "deliver_include";
    struct stat st;
    struct mypasswd *file_pwd = 0;
    int     status;
    VSTREAM *fp;
    int     fd;

    /*
     * Make verbose logging easier to understand.
     */
    state.level++;
    if (msg_verbose)
	MSG_LOG_STATE(myname, state);

    /*
     * DUPLICATE ELIMINATION
     * 
     * Don't process this include file more than once as this particular user.
     */
    if (been_here(state.dup_filter, "include %ld %s", (long) usr_attr.uid, path))
	return (0);
    state.msg_attr.exp_from = state.msg_attr.local;

    /*
     * Can of worms. Allow this include file to be symlinked, but disallow
     * inclusion of special files or of files with world write permission
     * enabled.
     */
    if (*path != '/') {
	msg_warn(":include:%s uses a relative path", path);
	dsb_simple(state.msg_attr.why, "5.3.5",
		   "mail system configuration error");
	return (bounce_append(BOUNCE_FLAGS(state.request),
			      BOUNCE_ATTR(state.msg_attr)));
    }
    if (stat_as(path, &st, usr_attr.uid, usr_attr.gid) < 0) {
	msg_warn("unable to lookup :include: file %s: %m", path);
	dsb_simple(state.msg_attr.why, "5.3.5",
		   "mail system configuration error");
	return (bounce_append(BOUNCE_FLAGS(state.request),
			      BOUNCE_ATTR(state.msg_attr)));
    }
    if (S_ISREG(st.st_mode) == 0) {
	msg_warn(":include: file %s is not a regular file", path);
	dsb_simple(state.msg_attr.why, "5.3.5",
		   "mail system configuration error");
	return (bounce_append(BOUNCE_FLAGS(state.request),
			      BOUNCE_ATTR(state.msg_attr)));
    }
    if (st.st_mode & S_IWOTH) {
	msg_warn(":include: file %s is world writable", path);
	dsb_simple(state.msg_attr.why, "5.3.5",
		   "mail system configuration error");
	return (bounce_append(BOUNCE_FLAGS(state.request),
			      BOUNCE_ATTR(state.msg_attr)));
    }

    /*
     * DELIVERY POLICY
     * 
     * Set the expansion type attribute so that we can decide if destinations
     * such as /file/name and |command are allowed at all.
     */
    state.msg_attr.exp_type = EXPAND_TYPE_INCL;

    /*
     * DELIVERY RIGHTS
     * 
     * When a non-root include file is listed in a root-owned alias, use the
     * rights of the include file owner.  We do not want to give the include
     * file owner control of the default account.
     * 
     * When an include file is listed in a user-owned alias or .forward file,
     * leave the delivery rights alone. Users should not be able to make
     * things happen with someone else's rights just by including some file
     * that is owned by their victim.
     */
    if (usr_attr.uid == 0) {
	if ((errno = mypwuid_err(st.st_uid, &file_pwd)) != 0 || file_pwd == 0) {
	    msg_warn(errno ? "cannot find username for uid %ld: %m" :
		     "cannot find username for uid %ld", (long) st.st_uid);
	    msg_warn("%s: cannot find :include: file owner username", path);
	    dsb_simple(state.msg_attr.why, "4.3.5",
		       "mail system configuration error");
	    return (defer_append(BOUNCE_FLAGS(state.request),
				 BOUNCE_ATTR(state.msg_attr)));
	}
	if (file_pwd->pw_uid != 0)
	    SET_USER_ATTR(usr_attr, file_pwd, state.level);
    }

    /*
     * MESSAGE FORWARDING
     * 
     * When no owner attribute is set (either via an owner- alias, or as part of
     * .forward file processing), set the owner attribute, to disable direct
     * delivery of local recipients. By now it is clear that the owner
     * attribute should have been called forwarder instead.
     */
    if (state.msg_attr.owner == 0)
	state.msg_attr.owner = state.msg_attr.rcpt.address;

    /*
     * From here on no early returns or we have a memory leak.
     * 
     * FILE OPEN RIGHTS
     * 
     * Use the delivery rights to open the include file. When no delivery rights
     * were established sofar, the file containing the :include: is owned by
     * root, so it should be OK to open any file that is accessible to root.
     * The command and file delivery routines are responsible for setting the
     * proper delivery rights. These are the rights of the default user, in
     * case the :include: is in a root-owned alias.
     * 
     * Don't propagate unmatched extensions unless permitted to do so.
     */
#define FOPEN_AS(p,u,g) ((fd = open_as(p,O_RDONLY,0,u,g)) >= 0 ? \
				vstream_fdopen(fd,O_RDONLY) : 0)

    if ((fp = FOPEN_AS(path, usr_attr.uid, usr_attr.gid)) == 0) {
	msg_warn("cannot open include file %s: %m", path);
	dsb_simple(state.msg_attr.why, "5.3.5",
		   "mail system configuration error");
	status = bounce_append(BOUNCE_FLAGS(state.request),
			       BOUNCE_ATTR(state.msg_attr));
    } else {
	if ((local_ext_prop_mask & EXT_PROP_INCLUDE) == 0)
	    state.msg_attr.unmatched = 0;
	close_on_exec(vstream_fileno(fp), CLOSE_ON_EXEC);
	status = deliver_token_stream(state, usr_attr, fp, (int *) 0);
	if (vstream_fclose(fp))
	    msg_warn("close %s: %m", path);
    }

    /*
     * Cleanup.
     */
    if (file_pwd)
	mypwfree(file_pwd);

    return (status);
}
