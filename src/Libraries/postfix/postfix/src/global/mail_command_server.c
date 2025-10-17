/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 4, 2022.
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
#include <stdlib.h>		/* 44BSD stdarg.h uses abort() */
#include <stdarg.h>
#include <string.h>

/* Utility library. */

#include <vstream.h>

/* Global library. */

#include "mail_proto.h"

/* mail_command_server - read single-command request */

int     mail_command_server(VSTREAM *stream,...)
{
    va_list ap;
    int     count;

    va_start(ap, stream);
    count = attr_vscan(stream, ATTR_FLAG_MISSING, ap);
    va_end(ap);
    return (count);
}
