/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 23, 2022.
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
#include <config.h>

#define VERSION_HIDDEN static

#include "roken.h"

#include "version.h"

void ROKEN_LIB_FUNCTION
print_version(const char *progname)
{
    const char *package_list = heimdal_version;

    if(progname == NULL)
	progname = getprogname();

    if(*package_list == '\0')
	package_list = "no version information";
    fprintf(stderr, "%s (%s)\n", progname, package_list);
    fprintf(stderr, "Copyright 1995-2011 Kungliga Tekniska HÃ¶gskolan\n");
#ifdef PACKAGE_BUGREPORT
    fprintf(stderr, "Send bug-reports to %s\n", PACKAGE_BUGREPORT);
#endif
}
