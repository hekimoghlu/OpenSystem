/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 16, 2023.
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

#include "roken.h"

ROKEN_LIB_FUNCTION ssize_t ROKEN_LIB_CALL
writev(int d, const struct iovec *iov, int iovcnt)
{
    ssize_t ret;
    size_t tot = 0;
    int i;
    char *buf, *p;

    for(i = 0; i < iovcnt; ++i)
	tot += iov[i].iov_len;
    buf = malloc(tot);
    if (tot != 0 && buf == NULL) {
	errno = ENOMEM;
	return -1;
    }
    p = buf;
    for (i = 0; i < iovcnt; ++i) {
	memcpy (p, iov[i].iov_base, iov[i].iov_len);
	p += iov[i].iov_len;
    }
    ret = write (d, buf, tot);
    free (buf);
    return ret;
}
