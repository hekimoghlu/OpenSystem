/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 24, 2025.
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
#include <unistd.h>

/* Utility library. */

#include "safe.h"

/* unsafe - can we trust user-provided environment, working directory, etc. */

int     unsafe(void)
{

    /*
     * The super-user is trusted.
     */
    if (getuid() == 0 && geteuid() == 0)
	return (0);

    /*
     * Danger: don't trust inherited process attributes, and don't leak
     * privileged info that the parent has no access to.
     */
    return (geteuid() != getuid()
#ifdef HAS_ISSETUGID
	    || issetugid()
#endif
	    || getgid() != getegid());
}
