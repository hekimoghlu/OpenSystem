/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 19, 2025.
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
#include "archive_platform.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include "archive.h"

struct write_memory_data {
	size_t	used;
	size_t  size;
	size_t * client_size;
	unsigned char * buff;
};

static int	memory_write_free(struct archive *, void *);
static int	memory_write_open(struct archive *, void *);
static ssize_t	memory_write(struct archive *, void *, const void *buff, size_t);

/*
 * Client provides a pointer to a block of memory to receive
 * the data.  The 'size' param both tells us the size of the
 * client buffer and lets us tell the client the final size.
 */
int
archive_write_open_memory(struct archive *a, void *buff, size_t buffSize, size_t *used)
{
	struct write_memory_data *mine;

	mine = (struct write_memory_data *)calloc(1, sizeof(*mine));
	if (mine == NULL) {
		archive_set_error(a, ENOMEM, "No memory");
		return (ARCHIVE_FATAL);
	}
	mine->buff = buff;
	mine->size = buffSize;
	mine->client_size = used;
	return (archive_write_open2(a, mine,
		    memory_write_open, memory_write, NULL, memory_write_free));
}

static int
memory_write_open(struct archive *a, void *client_data)
{
	struct write_memory_data *mine;
	mine = client_data;
	mine->used = 0;
	if (mine->client_size != NULL)
		*mine->client_size = mine->used;
	/* Disable padding if it hasn't been set explicitly. */
	if (-1 == archive_write_get_bytes_in_last_block(a))
		archive_write_set_bytes_in_last_block(a, 1);
	return (ARCHIVE_OK);
}

/*
 * Copy the data into the client buffer.
 * Note that we update mine->client_size on every write.
 * In particular, this means the client can follow exactly
 * how much has been written into their buffer at any time.
 */
static ssize_t
memory_write(struct archive *a, void *client_data, const void *buff, size_t length)
{
	struct write_memory_data *mine;
	mine = client_data;

	if (mine->used + length > mine->size) {
		archive_set_error(a, ENOMEM, "Buffer exhausted");
		return (ARCHIVE_FATAL);
	}
	memcpy(mine->buff + mine->used, buff, length);
	mine->used += length;
	if (mine->client_size != NULL)
		*mine->client_size = mine->used;
	return (length);
}

static int
memory_write_free(struct archive *a, void *client_data)
{
	struct write_memory_data *mine;
	(void)a; /* UNUSED */
	mine = client_data;
	if (mine == NULL)
		return (ARCHIVE_OK);
	free(mine);
	return (ARCHIVE_OK);
}
