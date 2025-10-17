/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 20, 2022.
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
__FBSDID("$FreeBSD: src/lib/libc/gen/getprogname.c,v 1.4 2002/03/29 22:43:41 markm Exp $");

#include "namespace.h"
#include <stdlib.h>
#include <crt_externs.h>
#define	__progname	(*_NSGetProgname())
#include "un-namespace.h"

#include "libc_private.h"

__weak_reference(_getprogname, getprogname);

const char *
_getprogname(void)
{

	return (__progname);
}
