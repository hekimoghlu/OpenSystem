/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 18, 2021.
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
/* System library. */

#include <sys_defs.h>

/* Utility library. */

#include <iostuff.h>
#include <name_mask.h>

int     unix_pass_fd_fix = 0;

/* set_unix_pass_fd_fix - set workaround programmatically */

void    set_unix_pass_fd_fix(const char *workarounds)
{
    const static NAME_MASK table[] = {
	"cmsg_len", UNIX_PASS_FD_FIX_CMSG_LEN,
	0,
    };

    unix_pass_fd_fix = name_mask("descriptor passing workarounds",
				 table, workarounds);
}
