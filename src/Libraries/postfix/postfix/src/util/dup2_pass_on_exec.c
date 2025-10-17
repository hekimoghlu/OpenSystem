/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 12, 2022.
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
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>

#define DO(s)	if (s < 0) { perror(#s); exit(1); }

int     main(int unused_argc, char **unused_argv)
{
    int     res;

    printf("Setting the close-on-exec flag of file-descriptor 0.\n");
    DO(fcntl(0, F_SETFD, 1));

    printf("Duplicating file-descriptor 0 to 3.\n");
    DO(dup2(0, 3));

    printf("Testing if the close-on-exec flag of file-descriptor 3 is set.\n");
    DO((res = fcntl(3, F_GETFD, 0)));
    if (res & 1)
	printf(
"Yes, a newly dup2()ed file-descriptor has the close-on-exec \
flag cloned.\n\
THIS VIOLATES Posix1003.1 section 6.2.1.2 or 6.5.2.2!\n\
You should #define DUP2_DUPS_CLOSE_ON_EXEC in sys_defs.h \
for your OS.\n");
    else
	printf(
"No, a newly dup2()ed file-descriptor has the close-on-exec \
flag cleared.\n\
This complies with Posix1003.1 section 6.2.1.2 and 6.5.2.2!\n");

    return 0;
}
