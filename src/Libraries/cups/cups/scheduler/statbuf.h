/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 23, 2022.
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
 * Constants...
 */

#define CUPSD_SB_BUFFER_SIZE	2048	/* Bytes for job status buffer */


/*
 * Types and structures...
 */

typedef struct				/**** Status buffer */
{
  int	fd;				/* File descriptor to read from */
  char	prefix[64];			/* Prefix for log messages */
  int	bufused;			/* How much is used in buffer */
  char	buffer[CUPSD_SB_BUFFER_SIZE];	/* Buffer */
} cupsd_statbuf_t;


/*
 * Prototypes...
 */

extern void		cupsdStatBufDelete(cupsd_statbuf_t *sb);
extern cupsd_statbuf_t	*cupsdStatBufNew(int fd, const char *prefix);
extern char		*cupsdStatBufUpdate(cupsd_statbuf_t *sb, int *loglevel,
			                    char *line, int linelen);
