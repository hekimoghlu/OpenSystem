/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 8, 2024.
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
__FBSDID("$FreeBSD: src/sbin/gpt/show.c,v 1.14.2.2.6.1 2010/02/10 00:26:20 kensmith Exp $");

#include <sys/types.h>

#include <err.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "map.h"
#include "gpt.h"

static int show_label = 0;
static int show_uuid = 0;

static void
usage_show(void)
{

	fprintf(stderr,
#ifdef __APPLE__
	    "usage: %s [-l] device ...\n", getprogname());
#else
	    "usage: %s [-lu] device ...\n", getprogname());
#endif
	exit(1);
}

static const char *
friendly(uuid_t *t)
{
#ifdef __APPLE__
	static char buf[40];

	uuid_unparse(*t, buf);

	return (buf);
#else
	static uuid_t boot = GPT_ENT_TYPE_FREEBSD_BOOT;
	static uuid_t efi_slice = GPT_ENT_TYPE_EFI;
	static uuid_t mslinux = GPT_ENT_TYPE_MS_BASIC_DATA;
	static uuid_t freebsd = GPT_ENT_TYPE_FREEBSD;
	static uuid_t hfs = GPT_ENT_TYPE_APPLE_HFS;
	static uuid_t apfs = GPT_ENT_TYPE_APPLE_APFS;
	static uuid_t linuxswap = GPT_ENT_TYPE_LINUX_SWAP;
	static uuid_t msr = GPT_ENT_TYPE_MS_RESERVED;
	static uuid_t swap = GPT_ENT_TYPE_FREEBSD_SWAP;
	static uuid_t ufs = GPT_ENT_TYPE_FREEBSD_UFS;
	static uuid_t vinum = GPT_ENT_TYPE_FREEBSD_VINUM;
	static uuid_t zfs = GPT_ENT_TYPE_FREEBSD_ZFS;
	static char buf[80];
	char *s;

	if (show_uuid)
		goto unfriendly;

	if (uuid_equal(t, &efi_slice, NULL))
		return ("EFI System");
	if (uuid_equal(t, &boot, NULL))
		return ("FreeBSD boot");
	if (uuid_equal(t, &swap, NULL))
		return ("FreeBSD swap");
	if (uuid_equal(t, &ufs, NULL))
		return ("FreeBSD UFS/UFS2");
	if (uuid_equal(t, &vinum, NULL))
		return ("FreeBSD vinum");
	if (uuid_equal(t, &zfs, NULL))
		return ("FreeBSD ZFS");

	if (uuid_equal(t, &freebsd, NULL))
		return ("FreeBSD legacy");
	if (uuid_equal(t, &mslinux, NULL))
		return ("Linux/Windows");
	if (uuid_equal(t, &linuxswap, NULL))
		return ("Linux swap");
	if (uuid_equal(t, &msr, NULL))
		return ("Windows reserved");
	if (uuid_equal(t, &hfs, NULL))
		return ("Apple HFS");
	if (uuid_equal(t, &apfs, NULL))
		return ("Apple APFS");
unfriendly:
	uuid_to_string(t, &s, NULL);
	strlcpy(buf, s, sizeof buf);
	free(s);
	return (buf);
#endif
}

static void
show(int fd __unused)
{
	uuid_t type;
	off_t start;
	map_t *m, *p;
	struct mbr *mbr;
	struct gpt_ent *ent;
	unsigned int i;

	printf("  %*s", lbawidth, "start");
	printf("  %*s", lbawidth, "size");
	printf("  index  contents\n");

	m = map_first();
	while (m != NULL) {
		printf("  %*llu", lbawidth, (long long)m->map_start);
		printf("  %*llu", lbawidth, (long long)m->map_size);
		putchar(' ');
		putchar(' ');
		if (m->map_index > 0)
			printf("%5d", m->map_index);
		else
			printf("     ");
		putchar(' ');
		putchar(' ');
		switch (m->map_type) {
		case MAP_TYPE_MBR:
			if (m->map_start != 0)
				printf("Extended ");
			printf("MBR");
			break;
		case MAP_TYPE_PRI_GPT_HDR:
			printf("Pri GPT header");
			break;
		case MAP_TYPE_SEC_GPT_HDR:
			printf("Sec GPT header");
			break;
		case MAP_TYPE_PRI_GPT_TBL:
			printf("Pri GPT table");
			break;
		case MAP_TYPE_SEC_GPT_TBL:
			printf("Sec GPT table");
			break;
		case MAP_TYPE_MBR_PART:
			p = m->map_data;
			if (p->map_start != 0)
				printf("Extended ");
			printf("MBR part ");
			mbr = p->map_data;
			for (i = 0; i < 4; i++) {
				start = le16toh(mbr->mbr_part[i].part_start_hi);
				start = (start << 16) +
				    le16toh(mbr->mbr_part[i].part_start_lo);
				if (m->map_start == p->map_start + start)
					break;
			}
			printf("%d", mbr->mbr_part[i].part_typ);
			break;
		case MAP_TYPE_GPT_PART:
			printf("GPT part ");
			ent = m->map_data;
			if (show_label) {
				printf("- \"%s\"",
				    utf16_to_utf8(ent->ent_name));
			} else {
				le_uuid_dec(&ent->ent_type, &type);
				printf("- %s", friendly(&type));
			}
			break;
		case MAP_TYPE_PMBR:
			printf("PMBR");
			break;
		}
		putchar('\n');
		m = m->map_next;
	}
}

int
cmd_show(int argc, char *argv[])
{
	int ch, fd;

	readonly = 1;

#ifdef __APPLE__
	while ((ch = getopt(argc, argv, "l")) != -1) {
#else
	while ((ch = getopt(argc, argv, "lu")) != -1) {
#endif
		switch(ch) {
		case 'l':
			show_label = 1;
			break;
		case 'u':
			show_uuid = 1;
			break;
		default:
			usage_show();
		}
	}

	if (argc == optind)
		usage_show();

	while (optind < argc) {
		fd = gpt_open(argv[optind++]);
		if (fd == -1) {
			warn("unable to open device '%s'", device_name);
			return (1);
		}

		show(fd);

		gpt_close(fd);
	}

	return (0);
}
