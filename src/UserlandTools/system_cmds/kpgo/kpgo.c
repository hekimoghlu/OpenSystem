/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 16, 2022.
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
#include <sys/pgo.h>

#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

static void usage(char **argv)
{
	fprintf (stderr, "usage: %s [-H] [-m] [-w] [uuid] >datafile\n", argv[0]);
        fprintf (stderr, "    uuid : the UUID of a kext\n");
        fprintf (stderr, "    -H   : grab data for the HIB segment\n");
        fprintf (stderr, "    -w   : wait for the kext to be unloaded\n");
        fprintf (stderr, "    -m   : request metadata\n");
        fprintf (stderr, "    -R   : reset all counters\n");
	exit(1);
}

int main(int argc, char **argv)
{
	int flags = 0;
        int data_flags = 0;
        uuid_t *uuidp = NULL;
        uuid_t uuid;
        int c;

        while ((c = getopt(argc, argv, "hHwmR")) != EOF) {
            switch(c) {
            case 'R':
                flags |= PGO_RESET_ALL;
                break;
            case 'H':
                flags |= PGO_HIB;
                break;
            case 'm':
                flags |= PGO_METADATA;
                break;
            case 'w':
                data_flags |= PGO_WAIT_FOR_UNLOAD;
                break;
            case '?':
            case 'h':
            default:
                usage(argv);
                break;
            }
        }

        if (optind < argc)
        {
            if (optind == argc - 1 &&
                0 == uuid_parse(argv[optind], uuid))
            {
                uuidp = &uuid;
            } else {
                usage(argv);
            }
        }

        if (flags & PGO_RESET_ALL) {
            if (flags != PGO_RESET_ALL || uuidp) {
                usage(argv);
            }
            ssize_t r = grab_pgo_data(NULL, PGO_RESET_ALL, NULL, 0);
            if (r < 0)
            {
                perror("grab_pgo_data");
                return 1;
            }
            return 0;
        }

	ssize_t size = grab_pgo_data(uuidp, flags, NULL, 0);

	if (size < 0)
	{
		perror("grab_pgo_data");
		return 1;
	}


	fprintf (stderr, "size = %ld\n", (long) size);

	unsigned char *buffer = valloc(size);
	if (!buffer)
	{
		perror("valloc");
		return 1;
	}

	ssize_t r = grab_pgo_data(uuidp, flags | data_flags, buffer, size);


	if (r < 0)
	{
		perror("grab_pgo_data");
		return 1;
	}

        if (isatty(STDOUT_FILENO)) {
            fprintf (stderr, "%s: refusing to write binary data to a tty!\n", argv[0]);
            return 1;
        }

        while (size > 0) {
            errno = 0;
            r = write(STDOUT_FILENO, buffer, size);
            if (r > 0) {
                buffer += r;
                size -= r;
            } else {
                perror ("write");
                return 1;
            }
        }

	return 0;
}
