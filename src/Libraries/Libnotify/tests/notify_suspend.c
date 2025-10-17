/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 10, 2024.
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

//
//  notify_suspend.c
//  Libnotify
//
//  Created by Brycen Wershing on 10/16/19.
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <notify.h>
#include <darwintest.h>
#include "notify_private.h"

static void run_test()
{
	__block bool suspendedToggle = false;
	__block bool notSuspendedToggle = false;
	int status;
	int suspendToken, noSuspendToken;
	const char *name = "com.apple.test.suspend";
	dispatch_queue_t queue = dispatch_queue_create("Test Q", NULL);

	status = notify_register_dispatch(name, &suspendToken, queue, ^(__unused int token){
		suspendedToggle = true;
	});
	T_ASSERT_EQ(status, NOTIFY_STATUS_OK, "Register suspend token");

	status = notify_register_dispatch(name, &noSuspendToken, queue, ^(__unused int token){
		notSuspendedToggle = true;
	});
	T_ASSERT_EQ(status, NOTIFY_STATUS_OK, "Register no-suspend token");

	sleep(1);

	T_ASSERT_EQ(suspendedToggle, false, "check suspend token before posts");
	T_ASSERT_EQ(notSuspendedToggle, false, "check no-suspend token before posts");

	notify_suspend(suspendToken);
	notify_post(name);
	sleep(1);

	T_ASSERT_EQ(suspendedToggle, false, "check suspend token after post");
	T_ASSERT_EQ(notSuspendedToggle, true, "check no-suspend token after post");

	suspendedToggle = false;
	notSuspendedToggle = false;
	notify_resume(suspendToken);
	sleep(1);

	T_ASSERT_EQ(suspendedToggle, true, "check suspend token after resume");
	T_ASSERT_EQ(notSuspendedToggle, false, "check no-suspend token after resume");

	notify_post(name);
	sleep(1);

	T_ASSERT_EQ(suspendedToggle, true, "check suspend token after final post");
	T_ASSERT_EQ(notSuspendedToggle, true, "check no-suspend token after final post");

	T_PASS("Test complete");
}

T_DECL(notify_suspend,
       "notify suspend registration test",
       T_META("owner", "Core Darwin Daemons & Tools"),
       T_META("as_root", "false"))
{
	run_test();
}

T_DECL(notify_suspend_filtered,
       "notify filtered suspend registration test",
       T_META("owner", "Core Darwin Daemons & Tools"),
       T_META("as_root", "false"))
{

	notify_set_options(NOTIFY_OPT_DISPATCH | NOTIFY_OPT_REGEN | NOTIFY_OPT_FILTERED);

	run_test();
}

