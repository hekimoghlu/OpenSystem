/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 18, 2022.
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
__FBSDID("$FreeBSD: src/lib/libc/gen/sysctlbyname.c,v 1.5 2002/02/01 00:57:29 obrien Exp $");

#include <sys/types.h>
#include <sys/sysctl.h>
#include <string.h>
#include <sys/errno.h>
#include <TargetConditionals.h>

#include "sysctl_internal.h"


int
sysctlbyname(const char *name, void *oldp, size_t *oldlenp, void *newp,
	     size_t newlen)
{
	int error;


	error = __sysctlbyname(name, strlen(name), oldp, oldlenp, newp, newlen);
	if (error < 0) {
		return error;
	}


	return error;
}

