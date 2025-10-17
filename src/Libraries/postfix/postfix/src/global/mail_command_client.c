/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 27, 2022.
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
#include <stdlib.h>			/* 44BSD stdarg.h uses abort() */
#include <stdarg.h>

/* Utility library. */

#include <vstream.h>
#include <msg.h>

/* Global library. */

#include <mail_proto.h>

/* mail_command_client - single-command transaction with completion status */

int     mail_command_client(const char *class, const char *name,...)
{
    va_list ap;
    VSTREAM *stream;
    int     status;

    /*
     * Talk a little protocol with the specified service.
     * 
     * This function is used for non-critical services where it is OK to back
     * off after the first error. Log what communication stage failed, to
     * facilitate trouble analysis.
     */
    if ((stream = mail_connect(class, name, BLOCKING)) == 0) {
	msg_warn("connect to %s/%s: %m", class, name);
	return (-1);
    }
    va_start(ap, name);
    status = attr_vprint(stream, ATTR_FLAG_NONE, ap);
    va_end(ap);
    if (status != 0) {
	msg_warn("write %s: %m", VSTREAM_PATH(stream));
	status = -1;
    } else if (attr_scan(stream, ATTR_FLAG_STRICT,
			 RECV_ATTR_INT(MAIL_ATTR_STATUS, &status), 0) != 1) {
	msg_warn("write/read %s: %m", VSTREAM_PATH(stream));
	status = -1;
    }
    (void) vstream_fclose(stream);
    return (status);
}
