/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 16, 2022.
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
#include <dispatch/dispatch.h>
#include <signal.h>
#include <sys/kern_memorystatus.h>
#include <sys/kern_memorystatus_freeze.h>

#include <darwintest.h>
#include <darwintest_utils.h>

#include "test_utils.h"

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.vm"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("VM"),
	T_META_CHECK_LEAKS(false),
	T_META_TAG_VM_PREFERRED
	);


T_HELPER_DECL(simple_bg, "no-op bg process") {
	signal(SIGUSR1, SIG_IGN);
	dispatch_source_t ds_signal = dispatch_source_create(DISPATCH_SOURCE_TYPE_SIGNAL, SIGUSR1, 0, dispatch_get_main_queue());
	if (ds_signal == NULL) {
		T_LOG("[fatal] dispatch source create failed.");
		exit(2);
	}

	dispatch_source_set_event_handler(ds_signal, ^{
		exit(0);
	});

	dispatch_activate(ds_signal);
	dispatch_main();
}

static pid_t helper_pid;
static void
signal_helper_process(void)
{
	kill(helper_pid, SIGUSR1);
}

T_DECL(memorystatus_disable_freeze_in_other_process, "memorystatus_disable_freezer for another process",
    T_META_BOOTARGS_SET("freeze_enabled=1"),
    T_META_REQUIRES_SYSCTL_EQ("vm.freeze_enabled", 1))
{
	helper_pid = launch_background_helper("simple_bg", true, true);
	T_ATEND(signal_helper_process);

	kern_return_t kern_ret = memorystatus_control(MEMORYSTATUS_CMD_SET_PROCESS_IS_FREEZABLE, helper_pid, 0, NULL, 0);
	T_QUIET; T_ASSERT_POSIX_SUCCESS(kern_ret, "set helper process as not freezable");
}
