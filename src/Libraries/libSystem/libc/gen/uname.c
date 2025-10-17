/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 21, 2025.
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
#include <sys/param.h>
#include <sys/sysctl.h>
#include <sys/utsname.h>

int
uname(name)
	struct utsname *name;
{
	int mib[2], rval;
	size_t len;
	char *p;

	rval = 0;

	mib[0] = CTL_KERN;
	mib[1] = KERN_OSTYPE;
	len = sizeof(name->sysname);
	if (sysctl(mib, 2, &name->sysname, &len, NULL, 0) == -1)
		rval = -1;

	mib[0] = CTL_KERN;
	mib[1] = KERN_HOSTNAME;
	len = sizeof(name->nodename);
	if (sysctl(mib, 2, &name->nodename, &len, NULL, 0) == -1)
		rval = -1;

	mib[0] = CTL_KERN;
	mib[1] = KERN_OSRELEASE;
	len = sizeof(name->release);
	if (sysctl(mib, 2, &name->release, &len, NULL, 0) == -1)
		rval = -1;

	/* The version may have newlines in it, turn them into spaces. */
	mib[0] = CTL_KERN;
	mib[1] = KERN_VERSION;
	len = sizeof(name->version);
	if (sysctl(mib, 2, &name->version, &len, NULL, 0) == -1)
		rval = -1;
	else
		for (p = name->version; len--; ++p)
			if (*p == '\n' || *p == '\t') {
				if (len > 1) {
					*p = ' ';
				} else {
					*p = '\0';
				}
			}
	mib[0] = CTL_HW;
	mib[1] = HW_MACHINE;
	len = sizeof(name->machine);
	if (sysctl(mib, 2, &name->machine, &len, NULL, 0) == -1)
		rval = -1;
	return (rval);
}
