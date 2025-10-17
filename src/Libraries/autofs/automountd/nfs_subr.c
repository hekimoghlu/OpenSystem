/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 12, 2024.
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
 *	nfs_subr.c
 *
 *	Copyright (c) 1996 Sun Microsystems Inc
 *	All Rights Reserved.
 */

#pragma ident	"@(#)nfs_subr.c	1.7	05/06/08 SMI"

#include <stdlib.h>
#include <string.h>
#include "nfs_subr.h"

#define fromhex(c)  ((c >= '0' && c <= '9') ? (c - '0') : \
	                ((c >= 'A' && c <= 'F') ? (c - 'A' + 10) :\
	                ((c >= 'a' && c <= 'f') ? (c - 'a' + 10) : 0)))

/*
 * The implementation of URLparse guarantees that the final string will
 * fit in the original one. Replaces '%' occurrences followed by 2 characters
 * with its corresponding hexadecimal character.
 */
void
URLparse(char *str)
{
	char *p, *q;

	p = q = str;
	while (*p) {
		*q = *p;
		if (*p++ == '%') {
			if (*p) {
				*q = fromhex(*p) * 16;
				p++;
				if (*p) {
					*q += fromhex(*p);
					p++;
				}
			}
		}
		q++;
	}
	*q = '\0';
}

/*
 * Convert from URL syntax to host:path syntax.
 */
int
convert_special(char **specialp, char *host, char *oldpath, char *newpath,
    char *cur_special)
{
	char *url;
	char *newspec;
	char *p;
	char *p1, *p2;
	int len;

	/*
	 * Rebuild the URL. This is necessary because parse replica
	 * assumes that nfs: is the host name.
	 */
	len = (int) (strlen("nfs:") + strlen(oldpath)) + 1;
	url = malloc(len);

	if (url == NULL) {
		return -1;
	}

	strlcpy(url, "nfs:", len);
	strlcat(url, oldpath, len);

	/*
	 * If we haven't done any conversion yet, allocate a buffer for it.
	 */
	if (*specialp == NULL) {
		newspec = *specialp = strdup(cur_special);
		if (newspec == NULL) {
			free(url);
			return -1;
		}
	} else {
		newspec = *specialp;
	}

	/*
	 * Now find the first occurence of the URL in the special string.
	 */
	p = strstr(newspec, url);

	if (p == NULL) {
		free(url);
		return -1;
	}

	p1 = p;
	p2 = host;

	/*
	 * Overwrite the URL in the special.
	 *
	 * Begin with the host name.
	 */
	for (;;) {
		/*
		 * Sine URL's take more room than host:path, there is
		 * no way we should hit a null byte in the original special.
		 */
		if (*p1 == '\0') {
			free(url);
			free(*specialp);
			*specialp = NULL;
			return -1;
		}

		if (*p2 == '\0') {
			break;
		}

		*p1 = *p2;
		p1++;
		p2++;
	}

	/*
	 * Add the : separator.
	 */
	*p1 = ':';
	p1++;

	/*
	 * Now over write into special the path portion of host:path in
	 */
	p2 = newpath;
	for (;;) {
		if (*p1 == '\0') {
			free(url);
			free(*specialp);
			*specialp = NULL;
			return -1;
		}
		if (*p2 == '\0') {
			break;
		}
		*p1 = *p2;
		p1++;
		p2++;
	}

	/*
	 * Now shift the rest of original special into the gap created
	 * by replacing nfs://host[:port]/path with host:path.
	 */
	p2 = p + strlen(url);
	for (;;) {
		if (*p1 == '\0') {
			free(url);
			free(*specialp);
			*specialp = NULL;
			return -1;
		}
		if (*p2 == '\0') {
			break;
		}
		*p1 = *p2;
		p1++;
		p2++;
	}

	*p1 = '\0';

	free(url);
	return 0;
}
