/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 23, 2024.
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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <notify.h>
#include <darwintest.h>

T_DECL(notify_many_dups,
       "notify many duplicate registration test",
       T_META("owner", "Core Darwin Daemons & Tools"),
       T_META("as_root", "false"),
       T_META_IGNORECRASHES("notify_many_dups*"))
{
	int t, n, i;
	uint32_t status;
	mach_port_t port = MACH_PORT_NULL;
	const char *name = "com.apple.notify.many.dups.test";

	n = 50000;

	status = notify_register_mach_port(name, &port, 0, &t);
	T_EXPECT_EQ_INT(status, NOTIFY_STATUS_OK, "notify_register_mach_port status == NOTIFY_STATUS_OK");
	for (i = 1; i < n; i++)
	{
		status = notify_register_mach_port(name, &port, NOTIFY_REUSE, &t);

		if (status != NOTIFY_STATUS_OK)  {
			T_FAIL("notify_register_mach_port status != NOTIFY_STATUS_OK (status: %d, iteration: %d", status, i);
		}
	}
	T_PASS("Successfully registered %d times for name %s\n", n, name);
}

#define notify_many_dups_number 500

T_DECL(notify_many_dups_posts,
       "notify many duplicate registration posting test",
       T_META("owner", "Core Darwin Daemons & Tools"),
       T_META("as_root", "false"),
       T_META_IGNORECRASHES("notify_many_dups*"))
{
	int i;
	uint32_t status;
	mach_port_t port = MACH_PORT_NULL;
	const char *name = "com.apple.notify.many.dups.post.test";

	int tokens[notify_many_dups_number];


	status = notify_register_mach_port(name, &port, 0, &tokens[0]);
	T_EXPECT_EQ_INT(status, NOTIFY_STATUS_OK, "notify_register_mach_port status == NOTIFY_STATUS_OK");
	for (i = 1; i < notify_many_dups_number; i++)
	{
		tokens[i] = NOTIFY_TOKEN_INVALID;
		status = notify_register_mach_port(name, &port, NOTIFY_REUSE, &tokens[i]);

		if (status != NOTIFY_STATUS_OK)  {
			T_FAIL("notify_register_mach_port status != NOTIFY_STATUS_OK (status: %d, iteration: %d)", status, i);
		}
		if (tokens[i] == NOTIFY_TOKEN_INVALID) {
			T_FAIL("notify_register_mach_port did not set token (iteration: %d)", i);
		}
	}
	T_PASS("Successfully registered %d times for name %s\n", notify_many_dups_number, name);

	notify_post(name);

	sleep(1);

	while (true) {
		int tid;
		mach_msg_empty_rcv_t msg;
		kern_return_t status;

		memset(&msg, 0, sizeof(msg));
		status = mach_msg(&msg.header, MACH_RCV_MSG | MACH_RCV_TIMEOUT, 0, sizeof(msg), port, 100, MACH_PORT_NULL);
		if (status != KERN_SUCCESS) {
			T_LOG("mach msg returned %d", status);
			break;
		}

		tid = msg.header.msgh_id;

		for (int j = 0; j < notify_many_dups_number; j++) {
			if (tokens[j] == tid) {
				T_PASS("Received message for registration %d (%d)", j, tokens[j]);
				tokens[j] = 0;
				notify_cancel(tid);
			}
		}
	}

	for (int k = 0; k < notify_many_dups_number; k++) {
		if (tokens[k] != 0) {
			notify_cancel(tokens[k]);
			T_FAIL("Did not receive notification for %d (%d)", k, tokens[k]);
		}
	}

}
