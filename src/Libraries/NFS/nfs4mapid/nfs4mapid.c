/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 18, 2024.
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
#include <stdint.h>
#include <stdio.h>
#include <strings.h>
#include <stdlib.h>
#include <unistd.h>
#include <err.h>
#include <sys/types.h>
#include <sys/mount.h>
#include <sys/ucred.h>
#include <sys/socket.h>
#include <sys/kauth.h>
#include <nfs/nfs.h>
#include <uuid/uuid.h>

#include "utils.h" // For load_nfs_kext()

char guid_grp_prefix[12] = "\xab\xcd\xef\xab\xcd\xef\xab\xcd\xef\xab\xcd\xef";

void
Usage(void)
{
	errx(1, "Usage: %s {-g gid | -u uid} | [-G] { name | guid}", getprogname());
}

void
print_map(struct nfs_testmapid *map)
{
	uuid_string_t guidstr;
	const char *type = map->ntm_grpflag ? "group" : "user";

	uuid_unparse(map->ntm_guid.g_guid, guidstr);

	switch (map->ntm_lookup) {
	case NTM_NAME2ID:
		printf("%s %s maps to id %d\n",
		    type, map->ntm_name, map->ntm_id);
		printf("\tmapping done through guid %s\n", guidstr);
		break;
	case NTM_ID2NAME:
		printf("%s id %d maps to %s\n", type, map->ntm_id, map->ntm_name);
		printf("\tmapping done through guid %s\n", guidstr);
		break;
	case NTM_NAME2GUID:
		printf("%s %s maps to guid %s\n", type, map->ntm_name, guidstr);
		break;
	case NTM_GUID2NAME:
		printf("%s guid %s maps to %s\n", type, guidstr, map->ntm_name);
		break;
	default:
		printf("Invalid mapping %d\n", map->ntm_lookup);
		break;
	}
}

#define NTM_INVALID ((uint32_t)-1)

int
main(int argc, char *argv[])
{
	int opt;
	char *eptr = NULL;
	struct nfs_testmapid map;
	int error;
	size_t namelen;
	const char *name;
	int grpflag = 0;

	memset(&map, 0, sizeof(map));
	map.ntm_lookup = NTM_INVALID;

	while ((opt = getopt(argc, argv, "u:g:G")) != -1) {
		switch (opt) {
		case 'g': map.ntm_grpflag = 1;
		case 'u': map.ntm_id = (uint32_t)strtoul(optarg, &eptr, 0);
			if (*eptr) {
				errx(1, "%s is not a valid uid/gid", optarg);
			}
			if (map.ntm_lookup != NTM_INVALID) {
				Usage();
			}
			map.ntm_lookup = NTM_ID2NAME;
			break;
		case 'G': map.ntm_grpflag = 1;
			if (map.ntm_lookup != NTM_INVALID) {
				Usage();
			}
			break;
		default:
			Usage();
			break;
		}
	}
	argc -= optind;
	argv += optind;
	name = *argv;

	if (map.ntm_lookup == NTM_INVALID) {
		if (argc != 1) {
			Usage();
		}
		if (uuid_parse(name, map.ntm_guid.g_guid)) {
			/* Not a guid */
			namelen = strnlen(*argv, MAXIDNAMELEN);
			if (namelen == MAXIDNAMELEN) {
				errx(1, "Passed in name is to long\n");
			}
			/* All "Well Known IDs" are groups */
			if (name[namelen - 1] == '@') {
				grpflag = 1;
				map.ntm_lookup = NTM_NAME2GUID;
			} else {
				map.ntm_lookup = NTM_NAME2ID;
			}
			strlcpy(map.ntm_name, *argv, MAXIDNAMELEN);
		} else {
			grpflag = (memcmp(guid_grp_prefix, map.ntm_guid.g_guid, sizeof(guid_grp_prefix)) == 0);
			map.ntm_lookup = NTM_GUID2NAME;
		}
	} else if (argc != 0) {
		Usage();
	}

	if (grpflag && !map.ntm_grpflag) {
		map.ntm_grpflag = 1;
		warnx("Setting '-G' option for known group GUID or name");
	}

	load_nfs_kext(); // Make sure nfs kext is UP!
	error = nfsclnt(NFSCLNT_TESTIDMAP, &map);
	if (error) {
		err(1, "nfsclnt failed");
	}

	print_map(&map);

	return 0;
}
