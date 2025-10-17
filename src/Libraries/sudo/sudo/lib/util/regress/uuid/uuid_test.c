/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 1, 2022.
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
#include <config.h>

#include <stdlib.h>
#if defined(HAVE_STDINT_H)
# include <stdint.h>
#elif defined(HAVE_INTTYPES_H)
# include <inttypes.h>
#endif
#include <string.h>
#include <unistd.h>

#define SUDO_ERROR_WRAP 0

#include "sudo_compat.h"
#include "sudo_fatal.h"
#include "sudo_util.h"

sudo_dso_public int main(int argc, char *argv[]);

/*
 * Test that sudo_uuid_create() generates a variant 1, version 4 uuid.
 */

/* From RFC 4122. */
struct uuid {
    uint32_t time_low;
    uint16_t time_mid;
    uint16_t time_hi_and_version;
    uint8_t clock_seq_hi_and_reserved;
    uint8_t clock_seq_low;
    uint8_t node[6];
};

int
main(int argc, char *argv[])
{
    int ch, errors = 0, ntests = 0;
    union {
        struct uuid id;
        unsigned char u8[16];
    } uuid;

    initprogname(argc > 0 ? argv[0] : "uuid_test");

    while ((ch = getopt(argc, argv, "v")) != -1) {
	switch (ch) {
	case 'v':
	    /* ignore */
	    break;
	default:
	    fprintf(stderr, "usage: %s [-v]\n", getprogname());
	    return EXIT_FAILURE;
	}
    }
    argc -= optind;
    argv += optind;

    /* Do 16 passes. */
    for (ntests = 0; ntests < 16; ntests++) {
	sudo_uuid_create(uuid.u8);

	/* Variant: two most significant bits (6 and 7) are 0 and 1. */
	if (ISSET(uuid.id.clock_seq_hi_and_reserved, (1 << 6))) {
	    sudo_warnx("uuid bit 6 set, should be clear");
	    errors++;
	    continue;
	}
	if (!ISSET(uuid.id.clock_seq_hi_and_reserved, (1 << 7))) {
	    sudo_warnx("uuid bit 7 clear, should be set");
	    errors++;
	    continue;
	}

	/* Version: bits 12-15 are 0010. */
	if ((uuid.id.time_hi_and_version & 0xf000) != 0x4000) {
	    sudo_warnx("bad version: 0x%x", uuid.id.time_hi_and_version & 0xf000);
	    errors++;
	    continue;
	}
    }

    if (ntests != 0) {
	printf("%s: %d tests run, %d errors, %d%% success rate\n",
	    getprogname(), ntests, errors, (ntests - errors) * 100 / ntests);
    }
    return errors;
}
