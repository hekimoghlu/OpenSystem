/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 18, 2022.
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
#include "includes.h"

#include <sys/types.h>

#include <stdarg.h>
#include <unistd.h>

#include "platform.h"

#include "openbsd-compat/openbsd-compat.h"

/*
 * Drop any fine-grained privileges that are not needed for post-startup
 * operation of ssh-agent
 *
 * Should be as close as possible to pledge("stdio cpath unix id proc exec", ...)
 */
void
platform_pledge_agent(void)
{
#ifdef USE_SOLARIS_PRIVS
	/*
	 * Note: Solaris priv dropping is closer to tame() than pledge(), but
	 * we will use what we have.
	 */
	solaris_drop_privs_root_pinfo_net();
#endif
}

/*
 * Drop any fine-grained privileges that are not needed for post-startup
 * operation of sftp-server
 */
void
platform_pledge_sftp_server(void)
{
#ifdef USE_SOLARIS_PRIVS
	solaris_drop_privs_pinfo_net_fork_exec();
#endif
}

/*
 * Drop any fine-grained privileges that are not needed for the post-startup
 * operation of the SSH client mux
 *
 * Should be as close as possible to pledge("stdio proc tty", ...)
 */
void
platform_pledge_mux(void)
{
#ifdef USE_SOLARIS_PRIVS
	solaris_drop_privs_root_pinfo_net_exec();
#endif
}
