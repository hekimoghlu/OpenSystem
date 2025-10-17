/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 6, 2021.
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
__FBSDID("$FreeBSD: src/sbin/gpt/recover.c,v 1.8.10.1 2010/02/10 00:26:20 kensmith Exp $");

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
usage_recover(void)
{

	fprintf(stderr,
	    "usage: %s device ...\n", getprogname());
	exit(1);
}

static int 
recover(int fd)
{
	off_t last;
	map_t *gpt, *tpg;
	map_t *tbl, *lbt;
	struct gpt_hdr *hdr;

	if (map_find(MAP_TYPE_MBR) != NULL) {
		warnx("%s: error: device contains a MBR", device_name);
		return 1;
	}

	gpt = map_find(MAP_TYPE_PRI_GPT_HDR);
	tpg = map_find(MAP_TYPE_SEC_GPT_HDR);
	tbl = map_find(MAP_TYPE_PRI_GPT_TBL);
	lbt = map_find(MAP_TYPE_SEC_GPT_TBL);

	if (gpt == NULL && tpg == NULL) {
		warnx("%s: no primary or secondary GPT headers, can't recover",
		    device_name);
		return 1;
	}
	if (tbl == NULL && lbt == NULL) {
		warnx("%s: no primary or secondary GPT tables, can't recover",
		    device_name);
		return 1;
	}

	last = mediasz / secsz - 1LL;

	if (tbl != NULL && lbt == NULL) {
		lbt = map_add(last - tbl->map_size, tbl->map_size,
		    MAP_TYPE_SEC_GPT_TBL, tbl->map_data);
		if (lbt == NULL) {
			warnx("%s: adding secondary GPT table failed",
			    device_name);
			return 1;
		}
		gpt_write(fd, lbt);
		warnx("%s: recovered secondary GPT table from primary",
		    device_name);
	} else if (tbl == NULL && lbt != NULL) {
		tbl = map_add(2LL, lbt->map_size, MAP_TYPE_PRI_GPT_TBL,
		    lbt->map_data);
		if (tbl == NULL) {
			warnx("%s: adding primary GPT table failed",
			    device_name);
			return 1;
		}
		gpt_write(fd, tbl);
		warnx("%s: recovered primary GPT table from secondary",
		    device_name);
	}

	if (gpt != NULL && tpg == NULL) {
		tpg = map_add(last, 1LL, MAP_TYPE_SEC_GPT_HDR,
		    calloc(1, secsz));
		if (tpg == NULL) {
			warnx("%s: adding secondary GPT header failed",
			    device_name);
			return 1;
		}
		memcpy(tpg->map_data, gpt->map_data, secsz);
		hdr = tpg->map_data;
		hdr->hdr_lba_self = htole64(tpg->map_start);
		hdr->hdr_lba_alt = htole64(gpt->map_start);
		hdr->hdr_lba_table = htole64(lbt->map_start);
		hdr->hdr_crc_self = 0;
		hdr->hdr_crc_self = htole32(crc32(hdr, le32toh(hdr->hdr_size)));
		gpt_write(fd, tpg);
		warnx("%s: recovered secondary GPT header from primary",
		    device_name);
	} else if (gpt == NULL && tpg != NULL) {
		gpt = map_add(1LL, 1LL, MAP_TYPE_PRI_GPT_HDR,
		    calloc(1, secsz));
		if (gpt == NULL) {
			warnx("%s: adding primary GPT header failed",
			    device_name);
			return 1;
		}
		memcpy(gpt->map_data, tpg->map_data, secsz);
		hdr = gpt->map_data;
		hdr->hdr_lba_self = htole64(gpt->map_start);
		hdr->hdr_lba_alt = htole64(tpg->map_start);
		hdr->hdr_lba_table = htole64(tbl->map_start);
		hdr->hdr_crc_self = 0;
		hdr->hdr_crc_self = htole32(crc32(hdr, le32toh(hdr->hdr_size)));
		gpt_write(fd, gpt);
		warnx("%s: recovered primary GPT header from secondary",
		    device_name);
	}
	return 0;
}

int
cmd_recover(int argc, char *argv[])
{
	int ch, fd;
	int ret = 0;

	while ((ch = getopt(argc, argv, "r")) != -1) {
		switch(ch) {
		case 'r':
			recoverable = 1;
			break;
		default:
			usage_recover();
		}
	}

	if (argc == optind)
		usage_recover();

	while (optind < argc) {
		fd = gpt_open(argv[optind++]);
		if (fd == -1) {
			warn("unable to open device '%s'", device_name);
			return (1);
		}

		ret = recover(fd);

		gpt_close(fd);
	}

	return (ret);
}
