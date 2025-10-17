/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 8, 2023.
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
/*
 * cover for getdomainname()
 * Copyright (C) 1995 by NeXT Computer, Inc.
 */
#include "libinfo_common.h"

#include <mach/mach_types.h>
#include <sys/sysctl.h>

LIBINFO_EXPORT
int
getdomainname(char *val, int inlen)
{
	int mib[2];
	size_t len = inlen;

	mib[0] = CTL_KERN;
	mib[1] = KERN_DOMAINNAME;
	return sysctl(mib, 2, val, &len, NULL, 0);
}
