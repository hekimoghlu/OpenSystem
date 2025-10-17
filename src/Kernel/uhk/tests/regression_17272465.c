/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 1, 2025.
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
#include <stdio.h>
#include <mach/mach.h>
#include <mach/host_priv.h>


T_DECL(regression_17272465,
    "Test for host_set_special_port Mach port over-release, rdr: 17272465", T_META_CHECK_LEAKS(false))
{
	kern_return_t kr;
	mach_port_t port = MACH_PORT_NULL;

	T_SETUPBEGIN;
	T_QUIET;
	T_ASSERT_MACH_SUCCESS(mach_port_allocate(mach_task_self(), MACH_PORT_RIGHT_RECEIVE, &port), NULL);
	T_QUIET;
	T_ASSERT_MACH_SUCCESS(mach_port_insert_right(mach_task_self(), port, port, MACH_MSG_TYPE_MAKE_SEND), NULL);
	T_SETUPEND;

	(void)host_set_special_port(mach_host_self(), 30, port);
	(void)host_set_special_port(mach_host_self(), 30, port);
	(void)host_set_special_port(mach_host_self(), 30, port);

	T_PASS("No panic occurred");
}
