/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 12, 2022.
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
__FBSDID("$FreeBSD: src/sbin/gpt/remove.c,v 1.10.10.1 2010/02/10 00:26:20 kensmith Exp $");

#include <sys/types.h>

#include <err.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "map.h"
#include "gpt.h"

static int all;
static uuid_t type;
static off_t block, size;
static unsigned int entry;

static void
usage_remove(void)
{

	fprintf(stderr,
	    "usage: %s -a device ...\n"
	    "       %s [-b lba] [-i index] [-s lba] [-t uuid] device ...\n",
	    getprogname(), getprogname());
	exit(1);
}

static int 
rem(int fd)
{
	uuid_t uuid;
	map_t *gpt, *tpg;
	map_t *tbl, *lbt;
	map_t *m;
	struct gpt_hdr *hdr;
	struct gpt_ent *ent;
	unsigned int i;

	gpt = map_find(MAP_TYPE_PRI_GPT_HDR);
	if (gpt == NULL) {
		warnx("%s: error: no primary GPT header; run create or recover",
		    device_name);
		return 1;
	}

	tpg = map_find(MAP_TYPE_SEC_GPT_HDR);
	if (tpg == NULL) {
		warnx("%s: error: no secondary GPT header; run recover",
		    device_name);
		return 1;
	}

	tbl = map_find(MAP_TYPE_PRI_GPT_TBL);
	lbt = map_find(MAP_TYPE_SEC_GPT_TBL);
	if (tbl == NULL || lbt == NULL) {
		warnx("%s: error: run recover -- trust me", device_name);
		return 1;
	}

	/* Remove all matching entries in the map. */
	for (m = map_first(); m != NULL; m = m->map_next) {
		if (m->map_type != MAP_TYPE_GPT_PART || m->map_index < 1)
			continue;
		if (entry > 0 && entry != m->map_index)
			continue;
		if (block > 0 && block != m->map_start)
			continue;
		if (size > 0 && size != m->map_size)
			continue;

		i = m->map_index - 1;

		hdr = gpt->map_data;
		ent = (void*)((char*)tbl->map_data + i *
		    le32toh(hdr->hdr_entsz));
		le_uuid_dec(&ent->ent_type, &uuid);
		if (!uuid_is_nil(&type, NULL) &&
		    !uuid_equal(&type, &uuid, NULL))
			continue;

		/* Remove the primary entry by clearing the partition type. */
		uuid_create_nil(&ent->ent_type, NULL);

		hdr->hdr_crc_table = htole32(crc32(tbl->map_data,
		    le32toh(hdr->hdr_entries) * le32toh(hdr->hdr_entsz)));
		hdr->hdr_crc_self = 0;
		hdr->hdr_crc_self = htole32(crc32(hdr, le32toh(hdr->hdr_size)));

		gpt_write(fd, gpt);
		gpt_write(fd, tbl);

		hdr = tpg->map_data;
		ent = (void*)((char*)lbt->map_data + i *
		    le32toh(hdr->hdr_entsz));

		/* Remove the secundary entry. */
		uuid_create_nil(&ent->ent_type, NULL);

		hdr->hdr_crc_table = htole32(crc32(lbt->map_data,
		    le32toh(hdr->hdr_entries) * le32toh(hdr->hdr_entsz)));
		hdr->hdr_crc_self = 0;
		hdr->hdr_crc_self = htole32(crc32(hdr, le32toh(hdr->hdr_size)));

		gpt_write(fd, lbt);
		gpt_write(fd, tpg);

#ifdef __APPLE__
		printf("%ss%u removed\n", device_name, m->map_index);
#else
		printf("%sp%u removed\n", device_name, m->map_index);
#endif
	}
	return 0;
}

int
cmd_remove(int argc, char *argv[])
{
	char *p;
	int ch, fd;
	int ret = 0;

	/* Get the remove options */
	while ((ch = getopt(argc, argv, "ab:i:s:t:")) != -1) {
		switch(ch) {
		case 'a':
			if (all > 0)
				usage_remove();
			all = 1;
			break;
		case 'b':
			if (block > 0)
				usage_remove();
			block = strtoll(optarg, &p, 10);
			if (*p != 0 || block < 1)
				usage_remove();
			break;
		case 'i':
			if (entry > 0)
				usage_remove();
			entry = strtol(optarg, &p, 10);
			if (*p != 0 || entry < 1)
				usage_remove();
			break;
		case 's':
			if (size > 0)
				usage_remove();
			size = strtoll(optarg, &p, 10);
			if (*p != 0 || size < 1)
				usage_remove();
			break;
		case 't':
			if (!uuid_is_nil(&type, NULL))
				usage_remove();
			if (parse_uuid(optarg, &type) != 0)
				usage_remove();
			break;
		default:
			usage_remove();
		}
	}

	if (!all ^
	    (block > 0 || entry > 0 || size > 0 || !uuid_is_nil(&type, NULL)))
		usage_remove();

	if (argc == optind)
		usage_remove();

	while (optind < argc) {
		fd = gpt_open(argv[optind++]);
		if (fd == -1) {
			warn("unable to open device '%s'", device_name);
			return (1);
		}

		ret = rem(fd);

		gpt_close(fd);
	}

	return (ret);
}
