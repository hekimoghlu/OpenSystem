/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 27, 2024.
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
 * replica.c
 *
 * Copyright 2005 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

/*
 * Parse replicated server lists of the form:
 *
 *	host1:/path1,host2,host3,host4:/path2,host5:/path3
 *
 * into an array containing its constituent parts:
 *
 *	host1	/path1
 *	host2	/path2
 *	host3	/path2
 *	host4	/path2
 *	host5	/path3
 * where a server could also be represented in form of literal address
 * and in case it is an IPv6 literal address it will be enclosed in
 * square brackets [IPv6 Literal address]
 * Problems indicated by null return; they will be memory allocation
 * errors worthy of an error message unless count == -1, which means
 * a parse error.
 */

#pragma ident	"@(#)replica.c	1.10	05/07/25 SMI"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/types.h>
#include <errno.h>
#include "replica.h"

void
free_replica(struct replica *list, int count)
{
	int i;

	for (i = 0; i < count; i++) {
		if (list[i].host) {
			free(list[i].host);
		}
		if (list[i].path) {
			free(list[i].path);
		}
	}
	free(list);
}

struct replica *
parse_replica(char *special, int *count)
{
	struct replica *list = NULL;
	char *root, *special2;
	char *proot, *x, *y;
	int scount, v6addr, i;
	int found_colon = 0;

	*count = 0;
	scount = 0;
	v6addr = 0;
	root = special2 = strdup(special);
	proot = root;

	while (root) {
		switch (*root) {
		case '[':
			if ((root != special2) && (*(root - 1) != ',')) {
				root++;
				break;
			}
			y = strchr(root, ']');
			if (!y) {
				root++;
				break;
			}
			if ((*(y + 1) != ',') && (*(y + 1) != ':')) {
				root = y + 1;
				break;
			}
			/*
			 * Found a v6 Literal Address, so set "v6addr"
			 * and grab the address and store it in the list
			 * under "host part".
			 */
			proot = root + 1;
			root = y + 1;
			v6addr = 1;
			if ((list = realloc(list, (*count + 1) *
			    sizeof(struct replica))) == NULL) {
				goto bad;
			}
			bzero(&list[(*count)++], sizeof(struct replica));
			*y = '\0';
			list[*count - 1].host = strdup(proot);
			if (!list[*count - 1].host) {
				goto bad;
			}
			break;
		case ':':
			*root = '\0';
			x = root + 1;
			/*
			 * Find comma (if present), which bounds the path.
			 * The comma implies that the user is trying to
			 * specify failover syntax if another colon follows.
			 */
			if (((y = strchr(x, ',')) != NULL) &&
			    (strchr((y + 1), ':'))) {
				root = y + 1;
				*y = '\0';
			} else {
				found_colon = 1;
				root = NULL;
			}
			/*
			 * If "v6addr" is set, unset it, and since the "host
			 * part" is already taken care of, skip to the "path
			 * path" part.
			 */
			if (v6addr == 1) {
				v6addr = 0;
			} else {
				if ((list = realloc(list, (*count + 1) *
				    sizeof(struct replica))) == NULL) {
					goto bad;
				}
				bzero(&list[(*count)++],
				    sizeof(struct replica));
				list[*count - 1].host = strdup(proot);
				if (!list[*count - 1].host) {
					goto bad;
				}
			}
			for (i = scount; i < *count; i++) {
				list[i].path = strdup(x);
				if (!list[i].path) {
					goto bad;
				}
			}
			scount = i;
			proot = root;
			if (y) {
				*y = ',';
			}
			break;
		case ',':
			/*
			 * If "v6addr" is set, unset it and continue
			 * else grab the address and store it in the list
			 * under "host part".
			 */
			if (v6addr == 1) {
				v6addr = 0;
				proot = ++root;
			} else {
				*root = '\0';
				root++;
				if ((list = realloc(list, (*count + 1) *
				    sizeof(struct replica))) == NULL) {
					goto bad;
				}
				bzero(&list[(*count)++],
				    sizeof(struct replica));
				list[*count - 1].host = strdup(proot);
				if (!list[*count - 1].host) {
					goto bad;
				}
				proot = root;
				*(root - 1) = ',';
			}
			break;
		default:
			if (*root == '\0') {
				root = NULL;
			} else {
				root++;
			}
		}
	}
	if (found_colon) {
		free(special2);
		return list;
	}
bad:
	if (list) {
		free_replica(list, *count);
	}
	if (!found_colon) {
		*count = -1;
	}
	free(special2);
	return NULL;
}
