/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 29, 2022.
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

#include <attr.h>

/* Global library. */

#include <mail_proto.h>
#include <dsn_print.h>

/* dsn_print - write DSN to stream */

int     dsn_print(ATTR_PRINT_MASTER_FN print_fn, VSTREAM *fp,
		          int flags, void *ptr)
{
    DSN    *dsn = (DSN *) ptr;
    int     ret;

    /*
     * The attribute order is determined by backwards compatibility. It can
     * be sanitized after all the ad-hoc DSN read/write code is replaced.
     */
    ret = print_fn(fp, flags | ATTR_FLAG_MORE,
		   SEND_ATTR_STR(MAIL_ATTR_DSN_STATUS, dsn->status),
		   SEND_ATTR_STR(MAIL_ATTR_DSN_DTYPE, dsn->dtype),
		   SEND_ATTR_STR(MAIL_ATTR_DSN_DTEXT, dsn->dtext),
		   SEND_ATTR_STR(MAIL_ATTR_DSN_MTYPE, dsn->mtype),
		   SEND_ATTR_STR(MAIL_ATTR_DSN_MNAME, dsn->mname),
		   SEND_ATTR_STR(MAIL_ATTR_DSN_ACTION, dsn->action),
		   SEND_ATTR_STR(MAIL_ATTR_WHY, dsn->reason),
		   ATTR_TYPE_END);
    return (ret);
}
