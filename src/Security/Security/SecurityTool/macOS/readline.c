/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 9, 2022.
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
#include "readline_cssm.h"
#include "security_tool.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <inttypes.h>

/* Read a line from stdin into buffer as a null terminated string.  If buffer is
   non NULL use at most buffer_size bytes and return a pointer to buffer.  Otherwise
   return a newly malloced buffer.
   if EOF is read this function returns NULL.  */
char *
readline(char *buffer, int buffer_size)
{
	int ix = 0, bytes_malloced = 0;

	if (!buffer)
	{
		bytes_malloced = 64;
		buffer = (char *)malloc(bytes_malloced);
		buffer_size = bytes_malloced;
	}

	for (;;++ix)
	{
		int ch;

		if (ix == buffer_size - 1)
		{
			if (!bytes_malloced)
				break;
			bytes_malloced += bytes_malloced;
			buffer = (char *)realloc(buffer, bytes_malloced);
			buffer_size = bytes_malloced;
		}

		ch = getchar();
		if (ch == EOF)
		{
			if (bytes_malloced)
				free(buffer);
			return NULL;
		}
		if (ch == '\n')
			break;
		buffer[ix] = ch;
	}

	/* 0 terminate buffer. */
	buffer[ix] = '\0';

	return buffer;
}

/* Read the file name into buffer.  On return buffer contains a newly
   malloced buffer or length buffer_size. Return 0 on success and -1 on failure.  */
int
read_file(const char *name, CSSM_DATA *outData)
{
	int fd = -1, result;
	char *buffer = NULL;
	off_t length;
	ssize_t bytes_read;

	do {
		fd = open(name, O_RDONLY, 0);
	} while (fd == -1 && errno == EINTR);

	if (fd == -1)
	{
		sec_error("open %s: %s", name, strerror(errno));
		result = -1;
		goto loser;
	}

	length = lseek(fd, 0, SEEK_END);
	if (length == -1)
	{
		sec_error("lseek %s, SEEK_END: %s", name, strerror(errno));
		result = -1;
		goto loser;
	}

	buffer = malloc((size_t)length);

	do {
		bytes_read = pread(fd, buffer, (size_t) length, 0);
	} while (bytes_read == -1 && errno == EINTR);

	if (bytes_read == -1)
	{
		sec_error("pread %s: %s", name, strerror(errno));
		result = -1;
		goto loser;
	}
	if (bytes_read != (ssize_t)length)
	{
        sec_error("read %s: only read %zd of %" PRId64 " bytes", name, bytes_read, (int64_t)length);
		result = -1;
		goto loser;
	}

    result = close(fd);
    if (result) {
        sec_error("close");
        goto loser;
    }

	outData->Data = (uint8 *)buffer;
	outData->Length = (uint32)length;

	return result;

loser:
    if (fd != -1) {
        close(fd);
    }
    if (buffer) {
        free(buffer);
    }

	return result;
}
