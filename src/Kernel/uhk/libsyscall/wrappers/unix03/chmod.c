/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 28, 2021.
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

#if __DARWIN_UNIX03

#include <sys/types.h>
#include <sys/stat.h>
#include "_errno.h"

extern int __chmod(const char *path, mode_t mode);

/*
 * chmod stub, ignore S_ISUID and/or S_ISGID on EPERM,
 * mandated for conformance.
 *
 * This is for UNIX03 only.
 */
int
chmod(const char *path, mode_t mode)
{
	int res = __chmod(path, mode);

	if (res >= 0 || errno != EPERM || (mode & (S_ISUID | S_ISGID)) == 0) {
		return res;
	}
	if (mode & S_ISGID) {
		res = __chmod(path, mode ^ S_ISGID);
		if (res >= 0 || errno != EPERM) {
			return res;
		}
	}
	if (mode & S_ISUID) {
		res = __chmod(path, mode ^ S_ISUID);
		if (res >= 0 || errno != EPERM) {
			return res;
		}
	}
	if ((mode & (S_ISUID | S_ISGID)) == (S_ISUID | S_ISGID)) {
		res = __chmod(path, mode ^ (S_ISUID | S_ISGID));
	}
	return res;
}

#endif /* __DARWIN_UNIX03 */
