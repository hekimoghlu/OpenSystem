/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 9, 2025.
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
/*
	File:		 cuFileIo.c 
	
	Description: simple file read/write utilities
*/

#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "cuFileIo.h"

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
	size_t			numBytes)
{
	int		rtn;
	int 	fd;
	
	fd = open(fileName, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if(fd == -1) {
		return errno;
	}
	rtn = (int)write(fd, bytes, (size_t)numBytes);
	if(rtn != (int)numBytes) {
		if(rtn >= 0) {
			printf("writeFile: short write\n");
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
int readFile(
	const char		*fileName,
	unsigned char	**bytes,		// mallocd and returned
	unsigned		*numBytes)		// returned
{
	int rtn;
	int fd;
	unsigned char *buf;
	struct stat	sb;
	unsigned size;
	
	*numBytes = 0;
	*bytes = NULL;
	fd = open(fileName, O_RDONLY, 0);
    if(fd == -1) {
		return errno;
	}
	rtn = fstat(fd, &sb);
	if(rtn) {
		goto errOut;
	}
	size = (unsigned)sb.st_size;
	buf = malloc(size);
	if(buf == NULL) {
		rtn = ENOMEM;
		goto errOut;
	}
	rtn = (int)lseek(fd, 0, SEEK_SET);
	if(rtn < 0) {
		free(buf);
		goto errOut;
	}
	rtn = (int)read(fd, buf, (size_t)size);
	if(rtn != (int)size) {
		if(rtn >= 0) {
			printf("readFile: short read\n");
		}
		free(buf);
		rtn = EIO;
	}
	else {
		rtn = 0;
		*bytes = buf;
		*numBytes = size;
	}
errOut:
	close(fd);
	return rtn;
}
