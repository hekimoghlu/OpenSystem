/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 11, 2025.
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
#include "ftp_locl.h"
RCSID("$Id$");

#if defined(KRB5)

void
afslog(int argc, char **argv)
{
    int ret;
    if(argc > 2) {
	printf("usage: %s [cell]\n", argv[0]);
	code = -1;
	return;
    }
    if(argc == 2)
	ret = command("SITE AFSLOG %s", argv[1]);
    else
	ret = command("SITE AFSLOG");
    code = (ret == COMPLETE);
}

#else
int ftp_afslog_placeholder;
#endif
