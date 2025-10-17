/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 14, 2023.
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
#include <darwintest.h>
#include <sys/wait.h>
#include <spawn.h>

T_GLOBAL_META(
	T_META_NAMESPACE("xnu.sysctl"),
	T_META_RADAR_COMPONENT_NAME("xnu"),
	T_META_RADAR_COMPONENT_VERSION("sysctl"),
	T_META_OWNER("p_tennen"),
	T_META_TAG_VM_PREFERRED,
	T_META_RUN_CONCURRENTLY(true)
	);

T_DECL(tree_walk, "Ensure we can walk a contrived sysctl tree")
{
	// rdar://138698424
	// Given a particular sysctl node tree (defined in-kernel)
	// When we invoke the sysctl machinery to walk this tree
	// (By specifying a partial path to the tree to the `sysctl` CLI tool -
	// trying to use sysctlbyname won't trigger the walk we're interested in.)
	char *args[] = { "/usr/sbin/sysctl", "debug.test.sysctl_node_test", NULL };
	int child_pid;
	T_ASSERT_POSIX_ZERO(posix_spawn(&child_pid, args[0], NULL, NULL, args, NULL), "posix_spawn() sysctl");
	// And we give the child a chance to execute
	int status = 0;
	T_ASSERT_POSIX_SUCCESS(waitpid(child_pid, &status, 0), "waitpid");
	// Then the machine does not panic :}
	T_PASS("The machine didn't panic, therefore our sysctl machinery can handle walking our node tree");
}
