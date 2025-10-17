/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 2, 2023.
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
#include <unistd.h>

#include "sudo_compat.h"
#include "sudo_util.h"

/*
 * Return a malloc()ed copy of the system hostname, or NULL if 
 * malloc() or gethostname() fails.
 */
char *
sudo_gethostname_v1(void)
{
    char *hname;
    size_t host_name_max;

#ifdef _SC_HOST_NAME_MAX
    host_name_max = (size_t)sysconf(_SC_HOST_NAME_MAX);
    if (host_name_max == (size_t)-1)
#endif
	host_name_max = 255;	/* POSIX and historic BSD */

    hname = malloc(host_name_max + 1);
    if (hname != NULL) {
	if (gethostname(hname, host_name_max + 1) == 0 && *hname != '\0') {
	    /* Old gethostname() may not NUL-terminate if there is no room. */
	    hname[host_name_max] = '\0';
	} else {
	    free(hname);
	    hname = NULL;
	}
    }
    return hname;
}
