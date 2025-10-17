/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 28, 2025.
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
//  notify_register_file_desc.c
//
//  Created by Brycen Wershing on 6/29/20.
//


#include <stdlib.h>
#include <notify.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <darwintest.h>
#include <signal.h>
#include "../libnotify.h"

T_DECL(notify_register_file_desc, "Make sure mach port registrations works",
		T_META("owner", "Core Darwin Daemons & Tools"),
		T_META_ASROOT(YES))
{
	const char *KEY = "com.apple.notify.test.file_desc";
	int rc, fd, tok;

	rc = notify_register_file_descriptor(KEY, &fd, 0, &tok);
	T_ASSERT_EQ(rc, NOTIFY_STATUS_OK, "register file desc should work");
	T_ASSERT_NE(fcntl(fd, F_GETFD), -1, "file descriptor should exist");

	rc = notify_cancel(tok);
	T_ASSERT_EQ(rc, NOTIFY_STATUS_OK, "cancel should work");
	T_ASSERT_EQ(fcntl(fd, F_GETFD), -1, "file descriptor should not exist");
}

