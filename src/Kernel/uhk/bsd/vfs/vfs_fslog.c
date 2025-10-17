/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 3, 2023.
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
#include <sys/errno.h>
#include <sys/types.h>
#include <kern/kalloc.h>
#include <sys/buf.h>
#include <sys/time.h>
#include <sys/kauth.h>
#include <sys/mount.h>
#include <sys/vnode.h>
#include <sys/syslog.h>
#include <sys/vnode_internal.h>
#include <sys/fslog.h>
#include <sys/mount_internal.h>
#if 0   /* see rdar://81514225 */
#include <sys/kasl.h>
#endif

#include <kern/zalloc.h>

#include <uuid/uuid.h>

#include <stdarg.h>

/* Log information about external modification of a process,
 * using MessageTracer formatting. Assumes that both the caller
 * and target are appropriately locked.
 * Currently prints following information -
 *      1. Caller process name (truncated to 16 characters)
 *	2. Caller process Mach-O UUID
 *  3. Target process name (truncated to 16 characters)
 *  4. Target process Mach-O UUID
 */
void
fslog_extmod_msgtracer(__unused proc_t caller, __unused proc_t target)
{
#if 0   /* see rdar://81514225 */
	if ((caller != PROC_NULL) && (target != PROC_NULL)) {
		/*
		 * Print into buffer large enough for "ThisIsAnApplicat(BC223DD7-B314-42E0-B6B0-C5D2E6638337)",
		 * including space for escaping, and NUL byte included in sizeof(uuid_string_t).
		 */

		uuid_string_t uuidstr;
		char c_name[2 * MAXCOMLEN + 2 /* () */ + sizeof(uuid_string_t)];
		char t_name[2 * MAXCOMLEN + 2 /* () */ + sizeof(uuid_string_t)];

		strlcpy(c_name, caller->p_comm, sizeof(c_name));
		uuid_unparse_upper(proc_executableuuid_addr(caller), uuidstr);
		strlcat(c_name, "(", sizeof(c_name));
		strlcat(c_name, uuidstr, sizeof(c_name));
		strlcat(c_name, ")", sizeof(c_name));
		if (0 != escape_str(c_name, strlen(c_name) + 1, sizeof(c_name))) {
			return;
		}

		strlcpy(t_name, target->p_comm, sizeof(t_name));
		uuid_unparse_upper(proc_executableuuid_addr(target), uuidstr);
		strlcat(t_name, "(", sizeof(t_name));
		strlcat(t_name, uuidstr, sizeof(t_name));
		strlcat(t_name, ")", sizeof(t_name));
		if (0 != escape_str(t_name, strlen(t_name) + 1, sizeof(t_name))) {
			return;
		}
#if DEBUG
		printf("EXTMOD: %s(%d) -> %s(%d)\n",
		    c_name,
		    proc_pid(caller),
		    t_name,
		    proc_pid(target));
#endif

		kern_asl_msg(LOG_DEBUG, "messagetracer",
		    5,
		    "com.apple.message.domain", "com.apple.kernel.external_modification",                                     /* 0 */
		    "com.apple.message.signature", c_name,                                     /* 1 */
		    "com.apple.message.signature2", t_name,                                     /* 2 */
		    "com.apple.message.result", "noop",                                     /* 3 */
		    "com.apple.message.summarize", "YES",                                     /* 4 */
		    NULL);
	}
#endif
}
