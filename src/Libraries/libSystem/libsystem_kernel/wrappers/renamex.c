/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 1, 2023.
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
#include <fcntl.h>

void __inc_remove_counter(void);
int __renameatx_np(int oldfd, const char *old, int newfd, const char *new, unsigned int flags);

int
renameatx_np(int oldfd, const char *old, int newfd, const char *new, unsigned int flags)
{
	int res = __renameatx_np(oldfd, old, newfd, new, flags);
	if (res == 0) {
		__inc_remove_counter();
	}
	return res;
}

int
renamex_np(const char *old, const char *new, unsigned int flags)
{
	return renameatx_np(AT_FDCWD, old, AT_FDCWD, new, flags);
}

// Deprecated
int
rename_ext(const char *old, const char *new, unsigned int flags)
{
	return renamex_np(old, new, flags);
}
