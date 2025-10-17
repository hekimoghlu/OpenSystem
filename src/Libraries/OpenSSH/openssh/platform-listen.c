/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 5, 2022.
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

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "log.h"
#include "misc.h"
#include "platform.h"

#include "openbsd-compat/openbsd-compat.h"

void
platform_pre_listen(void)
{
#ifdef LINUX_OOM_ADJUST
	/* Adjust out-of-memory killer so listening process is not killed */
	oom_adjust_setup();
#endif
}

void
platform_post_listen(void)
{
#ifdef SYSTEMD_NOTIFY
	ssh_systemd_notify_ready();
#endif
}

void
platform_pre_fork(void)
{
#ifdef USE_SOLARIS_PROCESS_CONTRACTS
	solaris_contract_pre_fork();
#endif
}

void
platform_pre_restart(void)
{
#ifdef SYSTEMD_NOTIFY
	ssh_systemd_notify_reload();
#endif
#ifdef LINUX_OOM_ADJUST
	oom_adjust_restore();
#endif
}

void
platform_post_fork_parent(pid_t child_pid)
{
#ifdef USE_SOLARIS_PROCESS_CONTRACTS
	solaris_contract_post_fork_parent(child_pid);
#endif
}

void
platform_post_fork_child(void)
{
#ifdef USE_SOLARIS_PROCESS_CONTRACTS
	solaris_contract_post_fork_child();
#endif
#ifdef LINUX_OOM_ADJUST
	oom_adjust_restore();
#endif
}

