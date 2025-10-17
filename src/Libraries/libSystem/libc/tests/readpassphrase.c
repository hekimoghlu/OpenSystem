/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 13, 2024.
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

#include <unistd.h>
#include <readpassphrase.h>

T_DECL(readpassphrase_stdin, "readpassphrase_stdin")
{
	int stdin_pipe[2] = { 0 };
	char pwd[] = "ishouldnotbedoingthis\n";
	char buff[128];

	T_ASSERT_POSIX_ZERO(pipe(stdin_pipe),
			"must be able to create a pipe");
	T_ASSERT_EQ(STDIN_FILENO, dup2(stdin_pipe[0], STDIN_FILENO),
			"must be able to re-register the read end of the pipe with STDIN_FILENO");
	T_ASSERT_EQ((ssize_t) sizeof(pwd), write(stdin_pipe[1], pwd, sizeof(pwd)),
			"must be able to write into the pipe");
	T_ASSERT_EQ((void *) buff, (void *) readpassphrase("", buff, sizeof(buff), RPP_STDIN),
			"readpassphrase must return its buffer argument on success");
	// readpassphrase eats newlines
	pwd[sizeof(pwd) - 2] = 0;
	T_ASSERT_EQ_STR(buff, pwd,
			"readpassphrase with RPP_STDIN must capture stdin");
}

