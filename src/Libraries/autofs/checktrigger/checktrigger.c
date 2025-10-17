/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 30, 2023.
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
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <sys/attr.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

struct attr_buffer {
	uint32_t        length;
	uint32_t        mount_flags;
};

int
main(int argc, char **argv)
{
	int i;
	struct attrlist attrs;
	struct attr_buffer attrbuf;

	if (argc < 2) {
		fprintf(stderr, "Usage: checktrigger <pathname>...\n");
		return 1;
	}
	argv++;
	argc--;
	for (i = 0; i < argc; i++) {
		memset(&attrs, 0, sizeof(attrs));
		attrs.bitmapcount = ATTR_BIT_MAP_COUNT;
		attrs.dirattr = ATTR_DIR_MOUNTSTATUS;
		if (getattrlist(argv[i], &attrs, &attrbuf, sizeof attrbuf,
		    FSOPT_NOFOLLOW) == -1) {
			fprintf(stderr, "checktrigger: getattrlist of %s failed: %s\n",
			    argv[i], strerror(errno));
			return 2;
		}
		printf("%s %s a trigger\n", argv[i],
		    (attrbuf.mount_flags & DIR_MNTSTATUS_TRIGGER) ?
		    "is" : "is not");
	}
	return 0;
}
