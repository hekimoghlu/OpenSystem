/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 16, 2025.
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
#ifndef _PRIVS_H
#define _PRIVS_H

#include <unistd.h>

/* Relinquish privileges temporarily for a setuid or setgid program
 * with the option of getting them back later.  This is done by
 * utilizing POSIX saved user and group IDs.  Call RELINQUISH_PRIVS once
 * at the beginning of the main program.  This will cause all operations
 * to be executed with the real userid.  When you need the privileges
 * of the setuid/setgid invocation, call PRIV_START; when you no longer
 * need it, call PRIV_END.  Note that it is an error to call PRIV_START
 * and not PRIV_END within the same function.
 *
 * Use RELINQUISH_PRIVS_ROOT(a,b) if your program started out running
 * as root, and you want to drop back the effective userid to a
 * and the effective group id to b, with the option to get them back
 * later.
 *
 * If you no longer need root privileges, but those of some other
 * userid/groupid, you can call REDUCE_PRIV(a,b) when your effective
 * is the user's.
 *
 * Problems: Do not use return between PRIV_START and PRIV_END; this
 * will cause the program to continue running in an unprivileged
 * state.
 *
 * It is NOT safe to call exec(), system() or popen() with a user-
 * supplied program (i.e. without carefully checking PATH and any
 * library load paths) with relinquished privileges; the called program
 * can acquire them just as easily.  Set both effective and real userid
 * to the real userid before calling any of them.
 */

extern uid_t real_uid, effective_uid;
extern gid_t real_gid, effective_gid;

#ifdef MAIN
uid_t real_uid, effective_uid;
gid_t real_gid, effective_gid;
#endif

#define RELINQUISH_PRIVS { \
	real_uid = getuid(); \
	effective_uid = geteuid(); \
	real_gid = getgid(); \
	effective_gid = getegid(); \
	if (seteuid(real_uid) != 0) err(1, "seteuid failed"); \
	if (setegid(real_gid) != 0) err(1, "setegid failed"); \
}

#define RELINQUISH_PRIVS_ROOT(a, b) { \
	real_uid = (a); \
	effective_uid = geteuid(); \
	real_gid = (b); \
	effective_gid = getegid(); \
	if (setegid(real_gid) != 0) err(1, "setegid failed"); \
	if (seteuid(real_uid) != 0) err(1, "seteuid failed"); \
}

#define PRIV_START { \
	if (seteuid(effective_uid) != 0) err(1, "seteuid failed"); \
	if (setegid(effective_gid) != 0) err(1, "setegid failed"); \
}

#define PRIV_END { \
	if (setegid(real_gid) != 0) err(1, "setegid failed"); \
	if (seteuid(real_uid) != 0) err(1, "seteuid failed"); \
}

#define REDUCE_PRIV(a, b) { \
	PRIV_START \
	effective_uid = (a); \
	effective_gid = (b); \
	if (setregid((gid_t)-1, effective_gid) != 0) err(1, "setregid failed"); \
	if (setreuid((uid_t)-1, effective_uid) != 0) err(1, "setreuid failed"); \
	PRIV_END \
}
#endif
