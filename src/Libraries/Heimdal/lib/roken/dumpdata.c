/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 11, 2024.
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

/*
 * Write datablob to a filename, don't care about errors.
 */

ROKEN_LIB_FUNCTION void ROKEN_LIB_CALL
rk_dumpdata (const char *filename, const void *buf, size_t size)
{
    int fd;

    fd = open(filename, O_WRONLY|O_TRUNC|O_CREAT, 0640);
    if (fd < 0)
	return;
    net_write(fd, buf, size);
    close(fd);
}

/*
 * Read all data from a filename, care about errors.
 */

ROKEN_LIB_FUNCTION int ROKEN_LIB_CALL
rk_undumpdata(const char *filename, void **buf, size_t *size)
{
    struct stat sb;
    int fd, ret;
    ssize_t sret;

    *buf = NULL;

    fd = open(filename, O_RDONLY, 0);
    if (fd < 0)
	return errno;
    if (fstat(fd, &sb) != 0){
	ret = errno;
	goto out;
    }
    if (sb.st_size > (off_t)(SIZE_T_MAX >> 1)) {
	ret = ERANGE;
	goto out;
    }
    *buf = malloc((size_t)sb.st_size);
    if (*buf == NULL) {
	ret = ENOMEM;
	goto out;
    }
    *size = (size_t)sb.st_size;

    sret = net_read(fd, *buf, *size);
    if (sret < 0)
	ret = errno;
    else if (sret != (ssize_t)*size) {
	ret = EINVAL;
	free(*buf);
	*buf = NULL;
    } else
	ret = 0;

 out:
    close(fd);
    return ret;
}
