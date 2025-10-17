/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 6, 2024.
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
 * This is an open source non-commercial project. Dear PVS-Studio, please check it.
 * PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
 */

#include <config.h>

#include <stdlib.h>
#include <string.h>

#include "sudo_compat.h"
#include "sudo_util.h"

/*
 * Declare/define __progname[] if necessary.
 * Assumes __progname[] is present if we have getprogname(3).
 */
#ifndef HAVE_SETPROGNAME
# if defined(HAVE_GETPROGNAME) || defined(HAVE___PROGNAME)
extern const char *__progname;
# else
static const char *__progname = "";
# endif /* HAVE_GETPROGNAME || HAVE___PROGNAME */
#endif /* HAVE_SETPROGNAME */

#ifndef HAVE_GETPROGNAME
const char *
sudo_getprogname(void)
{
    return __progname;
}
#endif

#ifndef HAVE_SETPROGNAME
void
sudo_setprogname(const char *name)
{
    __progname = sudo_basename(name);
}
#endif

void
initprogname2(const char *name, const char * const * allowed)
{
    const char *progname;
    int i;

    /* Fall back on "name" if getprogname() returns an empty string. */
    if ((progname = getprogname()) != NULL && *progname != '\0') {
	name = progname;
    } else {
	/* Make sure user-specified name is relative. */
	name = sudo_basename(name);
    }

    /* Check for libtool prefix and strip it if present. */
    if (name[0] == 'l' && name[1] == 't' && name[2] == '-' && name[3] != '\0')
	name += 3;

    /* Check allow list if present (first element is the default). */
    if (allowed != NULL) {
	for (i = 0; ; i++) {
	    if (allowed[i] == NULL) {
		name = allowed[0];
		break;
	    }
	    if (strcmp(allowed[i], name) == 0)
		break;
	}
    }

    /* Update internal progname if needed. */
    if (name != progname)
	setprogname(name);
    return;
}

void
initprogname(const char *name)
{
    initprogname2(name, NULL);
}
