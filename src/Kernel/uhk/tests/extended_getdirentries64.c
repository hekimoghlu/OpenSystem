/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 29, 2022.
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
#ifdef T_NAMESPACE
#undef T_NAMESPACE
#endif

#include <darwintest.h>
#include <darwintest_multiprocess.h>

#define PRIVATE 1
#include "../bsd/sys/dirent.h"

T_GLOBAL_META(T_META_RUN_CONCURRENTLY(true));

ssize_t __getdirentries64(int fd, void *buf, size_t bufsize, off_t *basep);

T_DECL(getdirentries64_extended, "check for GETDIRENTRIES64_EOF", T_META_TAG_VM_PREFERRED)
{
	char buf[GETDIRENTRIES64_EXTENDED_BUFSIZE];
	getdirentries64_flags_t *flags;
	ssize_t result;
	off_t offset;
	int fd;
	bool eof = false;

	flags = (getdirentries64_flags_t *)(uintptr_t)(buf + sizeof(buf) -
	    sizeof(getdirentries64_flags_t));
	fd = open("/", O_DIRECTORY | O_RDONLY);
	T_ASSERT_POSIX_SUCCESS(fd, "open(/)");

	for (;;) {
		*flags = (getdirentries64_flags_t)~0;
		result = __getdirentries64(fd, buf, sizeof(buf), &offset);
		T_ASSERT_POSIX_SUCCESS(result, "__getdirentries64()");
		T_ASSERT_LE((size_t)result, sizeof(buf) - sizeof(getdirentries64_flags_t),
		    "The kernel should have left space for the flags");
		T_ASSERT_NE(*flags, (getdirentries64_flags_t)~0,
		    "The kernel should have returned status");
		if (eof) {
			T_ASSERT_EQ(result, 0l, "At EOF, we really should be done");
			T_ASSERT_TRUE(*flags & GETDIRENTRIES64_EOF, "And EOF should still be set");
			T_END;
		}
		T_ASSERT_NE(result, 0l, "We're not at EOF, we should have an entry");
		eof = (*flags & GETDIRENTRIES64_EOF);
	}
}
