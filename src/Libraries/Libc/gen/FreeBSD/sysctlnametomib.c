/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 6, 2022.
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
__FBSDID("$FreeBSD: src/lib/libc/gen/sysctlnametomib.c,v 1.4 2003/01/04 00:11:11 tjr Exp $");

#include <sys/types.h>
#include <sys/sysctl.h>
#include <string.h>

extern int __sysctl(int *name, u_int namelen, void *oldp, size_t *oldlenp,
					void *newp, size_t newlen);

/*
 * This function uses a presently undocumented interface to the kernel
 * to walk the tree and get the type so it can print the value.
 * This interface is under work and consideration, and should probably
 * be killed with a big axe by the first person who can find the time.
 * (be aware though, that the proper interface isn't as obvious as it
 * may seem, there are various conflicting requirements.
 */
int
sysctlnametomib(const char *name, int *mibp, size_t *sizep)
{
	int oid[2];
	int error;

	oid[0] = 0;
	oid[1] = 3;

	*sizep *= sizeof (int);
	error = __sysctl(oid, 2, mibp, sizep, (void *)name, strlen(name));
	*sizep /= sizeof (int);
	return (error);
}
