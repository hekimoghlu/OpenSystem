/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 17, 2025.
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
#include <dsb_scan.h>

/* dsb_scan - read DSN_BUF from stream */

int     dsb_scan(ATTR_SCAN_MASTER_FN scan_fn, VSTREAM *fp,
		         int flags, void *ptr)
{
    DSN_BUF *dsb = (DSN_BUF *) ptr;
    int     ret;

    /*
     * The attribute order is determined by backwards compatibility. It can
     * be sanitized after all the ad-hoc DSN read/write code is replaced.
     */
    ret = scan_fn(fp, flags | ATTR_FLAG_MORE,
		  RECV_ATTR_STR(MAIL_ATTR_DSN_STATUS, dsb->status),
		  RECV_ATTR_STR(MAIL_ATTR_DSN_DTYPE, dsb->dtype),
		  RECV_ATTR_STR(MAIL_ATTR_DSN_DTEXT, dsb->dtext),
		  RECV_ATTR_STR(MAIL_ATTR_DSN_MTYPE, dsb->mtype),
		  RECV_ATTR_STR(MAIL_ATTR_DSN_MNAME, dsb->mname),
		  RECV_ATTR_STR(MAIL_ATTR_DSN_ACTION, dsb->action),
		  RECV_ATTR_STR(MAIL_ATTR_WHY, dsb->reason),
		  ATTR_TYPE_END);
    return (ret == 7 ? 1 : -1);
}
