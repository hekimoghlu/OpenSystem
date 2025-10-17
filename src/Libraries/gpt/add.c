/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 25, 2025.
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
__FBSDID("$FreeBSD: src/sbin/gpt/add.c,v 1.15.2.1.6.1 2010/02/10 00:26:20 kensmith Exp $");

#include <sys/types.h>

#include <err.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "map.h"
#include "gpt.h"

static uuid_t add_type;
static off_t add_block, add_size;
static unsigned int add_entry;

static void
usage_add(void)
{

	fprintf(stderr,
	    "usage: %s [-b lba] [-i index] [-s lba] [-t uuid] device ...\n",
	    getprogname());
	exit(1);
}

map_t *
gpt_add_part(int fd, uuid_t *type, off_t start, off_t size, unsigned int *entry)
{
	uuid_t uuid;
	map_t *gpt, *tpg;
	map_t *tbl, *lbt;
	map_t *map;
	struct gpt_hdr *hdr;
	struct gpt_ent *ent;
	unsigned int i;

	gpt = map_find(MAP_TYPE_PRI_GPT_HDR);
	ent = NULL;
	if (gpt == NULL) {
		warnx("%s: error: no primary GPT header; run create or recover",
		    device_name);
		return (NULL);
	}

	tpg = map_find(MAP_TYPE_SEC_GPT_HDR);
	if (tpg == NULL) {
		warnx("%s: error: no secondary GPT header; run recover",
		    device_name);
		return (NULL);
	}

	tbl = map_find(MAP_TYPE_PRI_GPT_TBL);
	lbt = map_find(MAP_TYPE_SEC_GPT_TBL);
	if (tbl == NULL || lbt == NULL) {
		warnx("%s: error: run recover -- trust me", device_name);
		return (NULL);
	}

	hdr = gpt->map_data;
	if (*entry > le32toh(hdr->hdr_entries)) {
		warnx("%s: error: index %u out of range (%u max)", device_name,
		    *entry, le32toh(hdr->hdr_entries));
		return (NULL);
	}

	if (*entry > 0) {
		i = *entry - 1;
		ent = (void*)((char*)tbl->map_data + i *
		    le32toh(hdr->hdr_entsz));
		if (!uuid_is_nil(&ent->ent_type, NULL)) {
			warnx("%s: error: entry at index %u is not free",
			    device_name, *entry);
			return (NULL);
		}
	} else {
		/* Find empty slot in GPT table. */
		for (i = 0; i < le32toh(hdr->hdr_entries); i++) {
			ent = (void*)((char*)tbl->map_data + i *
			    le32toh(hdr->hdr_entsz));
			if (uuid_is_nil(&ent->ent_type, NULL))
				break;
		}
		if (i == le32toh(hdr->hdr_entries)) {
			warnx("%s: error: no available table entries",
			    device_name);
			return (NULL);
		}
	}

	map = map_alloc(start, size);
	if (map == NULL) {
		warnx("%s: error: no space available on device", device_name);
		return (NULL);
	}

	le_uuid_dec(&ent->ent_uuid, &uuid);
	if (uuid_is_nil(&uuid, NULL)) {
		uuid_create(&uuid, NULL);
	}

	le_uuid_enc(&ent->ent_type, type);
	le_uuid_enc(&ent->ent_uuid, &uuid);
	ent->ent_lba_start = htole64(map->map_start);
	ent->ent_lba_end = htole64(map->map_start + map->map_size - 1LL);

	hdr->hdr_crc_table = htole32(crc32(tbl->map_data,
	    le32toh(hdr->hdr_entries) * le32toh(hdr->hdr_entsz)));
	hdr->hdr_crc_self = 0;
	hdr->hdr_crc_self = htole32(crc32(hdr, le32toh(hdr->hdr_size)));

	gpt_write(fd, gpt);
	gpt_write(fd, tbl);

	hdr = tpg->map_data;
	ent = (void*)((char*)lbt->map_data + i * le32toh(hdr->hdr_entsz));

	le_uuid_enc(&ent->ent_type, type);
	le_uuid_enc(&ent->ent_uuid, &uuid);
	ent->ent_lba_start = htole64(map->map_start);
	ent->ent_lba_end = htole64(map->map_start + map->map_size - 1LL);

	hdr->hdr_crc_table = htole32(crc32(lbt->map_data,
	    le32toh(hdr->hdr_entries) * le32toh(hdr->hdr_entsz)));
	hdr->hdr_crc_self = 0;
	hdr->hdr_crc_self = htole32(crc32(hdr, le32toh(hdr->hdr_size)));

	gpt_write(fd, lbt);
	gpt_write(fd, tpg);

	*entry = i + 1;

	return (map);
}

static int 
add(int fd)
{

	if (!gpt_add_part(fd, &add_type, add_block, add_size, &add_entry))
		return (1);

#ifdef __APPLE__
	printf("%ss%u added\n", device_name, add_entry);
#else
	printf("%sp%u added\n", device_name, add_entry);
#endif
		return (0);
}

int
cmd_add(int argc, char *argv[])
{
	char *p;
	int ch, fd;
	int ret = 0;

	/* Get the migrate options */
	while ((ch = getopt(argc, argv, "b:i:s:t:")) != -1) {
		switch(ch) {
		case 'b':
			if (add_block > 0)
				usage_add();
			add_block = strtoll(optarg, &p, 10);
			if (*p != 0 || add_block < 1)
				usage_add();
			break;
		case 'i':
			if (add_entry > 0)
				usage_add();
			add_entry = strtol(optarg, &p, 10);
			if (*p != 0 || add_entry < 1)
				usage_add();
			break;
		case 's':
			if (add_size > 0)
				usage_add();
			add_size = strtoll(optarg, &p, 10);
			if (*p != 0 || add_size < 1)
				usage_add();
			break;
		case 't':
			if (!uuid_is_nil(&add_type, NULL))
				usage_add();
			if (parse_uuid(optarg, &add_type) != 0)
				usage_add();
			break;
		default:
			usage_add();
		}
	}

	if (argc == optind)
		usage_add();

#ifdef __APPLE__
	/* Create HFS partitions by default. */
	if (uuid_is_null(add_type))
		uuid_copy(add_type, GPT_ENT_TYPE_APPLE_HFS);
#else
	/* Create UFS partitions by default. */
	if (uuid_is_nil(&add_type, NULL)) {
		uuid_t ufs = GPT_ENT_TYPE_FREEBSD_UFS;
		add_type = ufs;
	}
#endif

	while (optind < argc) {
		fd = gpt_open(argv[optind++]);
		if (fd == -1) {
			warn("unable to open device '%s'", device_name);
			return (1);
		}

		ret = add(fd);

		gpt_close(fd);
	}

	return (ret);
}
