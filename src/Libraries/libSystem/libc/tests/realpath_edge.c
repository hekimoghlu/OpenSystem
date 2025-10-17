/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 17, 2022.
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
#include <sys/cdefs.h>

#include <sys/param.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <darwintest.h>

// Required for ASan
#include "../fbsdcompat/_fbsd_compat_.h"
#include "../stdlib/FreeBSD/realpath.c"
#include "../gen/FreeBSD/getcwd.c"


T_DECL(realpath_buffer_overflow, "Test for out of bounds read from 'left' array (compile realpath.c with '-fsanitize=address')")
{
	char path[MAXPATHLEN] = { 0 };
	char resb[MAXPATHLEN] = { 0 };
	size_t i;

	path[0] = 'a';
	path[1] = '/';
	for (i = 2; i < sizeof(path) - 1; ++i) {
		path[i] = 'a';
	}

	T_ASSERT_NULL(realpath(path, resb), NULL);
}

