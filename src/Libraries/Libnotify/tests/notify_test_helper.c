/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 2, 2021.
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
//  notify_exec_helper.c
//  darwintests
//
//  Created by Brycen Wershing on 2/10/20.
//

#include <stdio.h>
#include <stdlib.h>
#include <notify.h>
#include <unistd.h>
 #include <errno.h>

extern char **environ;

#define KEY "com.apple.notify.test"

int main(int argc, char *argv[]) {

	printf("In notify_test_helper\n");

	uint32_t rc;
	int c_token, d_token;
	int check;
	__block bool dispatch_called = false;

	dispatch_queue_t dq = dispatch_queue_create_with_target("q", NULL, NULL);
	
	rc = notify_register_check(KEY, &c_token);
	if (rc != NOTIFY_STATUS_OK)
	{
		return 51;
	}

	notify_check(c_token, &check);
	if (check != 1) {
		return 52;
	}
	notify_check(c_token, &check);
	if (check != 0) {
		return 53;
	}

	rc = notify_register_dispatch(KEY, &d_token, dq, ^(int token){
		dispatch_called = true;
	});
	if (rc != NOTIFY_STATUS_OK)
	{
		return 54;
	}

	sleep(1);

	if (dispatch_called) {
		return 55;
	}

	rc = notify_post(KEY);
	if (rc != NOTIFY_STATUS_OK)
	{
		return 56;
	}

	sleep(1);

	notify_check(c_token, &check);
	if (check != 1) {
		return 57;
	}

	if (!dispatch_called) {
		return 58;
	}

	rc = notify_cancel(c_token);
	if (rc != NOTIFY_STATUS_OK)
	{
		return 59;
	}

	rc = notify_cancel(d_token);
	if (rc != NOTIFY_STATUS_OK)
	{
		return 60;
	}

	if (argc == 2) {
		printf("Calling exec again\n");

		char *argv[2];
		argv[0] = "notify_test_helper";
		argv[1] = NULL;

		execve("/AppleInternal/Tests/Libnotify/notify_test_helper" ,argv , environ);
		printf("execve failed with %d\n", errno);
		abort();
	}

	printf("Succeeded!\n");

	return 42;
}
