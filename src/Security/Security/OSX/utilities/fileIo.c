/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 3, 2025.
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
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdint.h>
#include "fileIo.h"

int writeFile(
              const char			*fileName,
              const unsigned char	*bytes,
              unsigned              numBytes)
{
    size_t n = numBytes;
    return writeFileSizet(fileName, bytes, n);
}

int writeFileSizet(
	const char			*fileName,
	const unsigned char	*bytes,
	size_t              numBytes)
{
	int		rtn;
	int 	fd;
    ssize_t wrc;

    if (!fileName) {
        fwrite(bytes, 1, numBytes, stdout);
        fflush(stdout);
        return ferror(stdout);
    }

	fd = open(fileName, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if(fd == -1) {
		return errno;
	}
	wrc = write(fd, bytes, (size_t)numBytes);
	if(wrc != (ssize_t) numBytes) {
		if(wrc >= 0) {
			fprintf(stderr, "writeFile: short write\n");
		}
		rtn = EIO;
	}
	else {
		rtn = 0;
	}
	close(fd);
	return rtn;
}

/*
 * Read entire file.
 */
int readFileSizet(
	const char		*fileName,
	unsigned char	**bytes,		// mallocd and returned
	size_t          *numBytes)		// returned
{
	int rtn;
	int fd;
	char *buf;
	struct stat	sb;
	size_t size;
    ssize_t rrc;

	*numBytes = 0;
	*bytes = NULL;
	fd = open(fileName, O_RDONLY);
    if(fd == -1) {
		return errno;
	}
	rtn = fstat(fd, &sb);
	if(rtn) {
		goto errOut;
	}
	if (sb.st_size > (off_t) ((UINT32_MAX >> 1)-1)) {
		rtn = EFBIG;
		goto errOut;
	}
	size = (size_t)sb.st_size;
	buf = (char *)malloc(size);
	if(buf == NULL) {
		rtn = ENOMEM;
		goto errOut;
	}
	rrc = read(fd, buf, size);
	if(rrc != (ssize_t) size) {
		if(rtn >= 0) {
            free(buf);
			fprintf(stderr, "readFile: short read\n");
		}
		rtn = EIO;
	}
	else {
		rtn = 0;
		*bytes = (unsigned char *)buf;
		*numBytes = size;
	}

errOut:
	close(fd);
	return rtn;
}
