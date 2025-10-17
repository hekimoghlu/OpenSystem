/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 6, 2023.
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
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/sbin/gpt/destroy.c,v 1.6.10.1 2010/02/10 00:26:20 kensmith Exp $");

#include <sys/types.h>

#include <err.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "map.h"
#include "gpt.h"

static int recoverable;

static void
usage_destroy(void)
{

	fprintf(stderr,
	    "usage: %s [-r] device ...\n", getprogname());
	exit(1);
}

static int 
destroy(int fd)
{
	map_t *pri_hdr, *sec_hdr;
	map_t *pmbr;

	pri_hdr = map_find(MAP_TYPE_PRI_GPT_HDR);
	sec_hdr = map_find(MAP_TYPE_SEC_GPT_HDR);
	pmbr = map_find(MAP_TYPE_PMBR);

	if (pri_hdr == NULL && sec_hdr == NULL) {
		warnx("%s: error: device doesn't contain a GPT", device_name);
		return 1;
	}

	if (recoverable && sec_hdr == NULL) {
		warnx("%s: error: recoverability not possible", device_name);
		return 1;
	}

	if (pri_hdr != NULL) {
		bzero(pri_hdr->map_data, secsz);
		gpt_write(fd, pri_hdr);
	}

	if (!recoverable && sec_hdr != NULL) {
		bzero(sec_hdr->map_data, secsz);
		gpt_write(fd, sec_hdr);
	}

	if (!recoverable && pmbr != NULL) {
		bzero(pmbr->map_data, secsz);
		gpt_write(fd, pmbr);
	}
        return 0;
}

int
cmd_destroy(int argc, char *argv[])
{
	int ch, fd;
	int ret = 0;

	while ((ch = getopt(argc, argv, "r")) != -1) {
		switch(ch) {
		case 'r':
			recoverable = 1;
			break;
		default:
			usage_destroy();
		}
	}

	if (argc == optind)
		usage_destroy();

	while (optind < argc) {
		fd = gpt_open(argv[optind++]);
		if (fd == -1) {
			warn("unable to open device '%s'", device_name);
			return (1);
		}

		ret = destroy(fd);

		gpt_close(fd);
	}

	return (ret);
}
