/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 28, 2022.
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
#ifndef NO_SYSCALL_LEGACY

#define _NONSTD_SOURCE
#include <sys/cdefs.h>

/*
 * We need conformance on so that EOPNOTSUPP=102.  But the routine symbol
 * will still be the legacy (undecorated) one.
 */
#undef __DARWIN_UNIX03
#define __DARWIN_UNIX03 1

#include <sys/attr.h>
#include "_errno.h"

#ifdef __LP64__
extern int __setattrlist(const char *, void *, void *, size_t, unsigned int);
#else /* !__LP64__ */
extern int __setattrlist(const char *, void *, void *, size_t, unsigned long);
#endif /* __LP64__ */

/*
 * setattrlist stub, legacy version
 */
int
#ifdef __LP64__
setattrlist(const char *path, void *attrList, void *attrBuf,
    size_t attrBufSize, unsigned int options)
#else /* !__LP64__ */
setattrlist(const char *path, void *attrList, void *attrBuf,
    size_t attrBufSize, unsigned long options)
#endif /* __LP64__ */
{
	int ret = __setattrlist(path, attrList, attrBuf, attrBufSize, options);

	/* use ENOTSUP for legacy behavior */
	if (ret < 0 && errno == EOPNOTSUPP) {
		errno = ENOTSUP;
	}
	return ret;
}

#endif
