/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 10, 2021.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "sudo_compat.h"
#include "sudo_fatal.h"
#include "sudo_queue.h"
#include "sudo_digest.h"
#include "sudo_util.h"
#include "parse.h"

sudo_dso_public int main(int argc, char *argv[]);

#define NUM_TESTS	8
static const char *test_strings[NUM_TESTS] = {
    "",
    "a",
    "abc",
    "message digest",
    "abcdefghijklmnopqrstuvwxyz",
    "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
    "12345678901234567890123456789012345678901234567890123456789"
	"012345678901234567890",
};

static unsigned char *
check_digest(int digest_type, const char *buf, size_t buflen, size_t *digest_len)
{
    char tfile[] = "digest.XXXXXX";
    unsigned char *digest = NULL;
    int tfd;

    /* Write test data to temporary file. */
    tfd = mkstemp(tfile);
    if (tfd == -1) {
	sudo_warn_nodebug("mkstemp");
	goto done;
    }
    if ((size_t)write(tfd, buf, buflen) != buflen) {
	sudo_warn_nodebug("write");
	goto done;
    }
    lseek(tfd, 0, SEEK_SET);

    /* Get file digest. */
    digest = sudo_filedigest(tfd, tfile, digest_type, digest_len);
    if (digest == NULL) {
	/* Warning (if any) printed by sudo_filedigest() */
	goto done;
    }
done:
    if (tfd != -1) {
	close(tfd);
	unlink(tfile);
    }
    return digest;
}

int
main(int argc, char *argv[])
{
    static const char hex[] = "0123456789abcdef";
    char buf[1000 * 1000];
    unsigned char *digest;
    unsigned int i, j;
    size_t digest_len;
    int digest_type;

    initprogname(argc > 0 ? argv[0] : "check_digest");

    for (digest_type = 0; digest_type < SUDO_DIGEST_INVALID; digest_type++) {
	for (i = 0; i < NUM_TESTS; i++) {
	    digest = check_digest(digest_type, test_strings[i],
		strlen(test_strings[i]), &digest_len);
	    if (digest != NULL) {
		printf("%s (\"%s\") = ", digest_type_to_name(digest_type),
		    test_strings[i]);
		for (j = 0; j < digest_len; j++) {
		    putchar(hex[digest[j] >> 4]);
		    putchar(hex[digest[j] & 0x0f]);
		}
		putchar('\n');
		free(digest);
	    }
	}

	/* Simulate a string of a million 'a' characters. */
	memset(buf, 'a', sizeof(buf));
	digest = check_digest(digest_type, buf, sizeof(buf), &digest_len);
	if (digest != NULL) {
	    printf("%s (one million 'a' characters) = ",
		digest_type_to_name(digest_type));
	    for (j = 0; j < digest_len; j++) {
		putchar(hex[digest[j] >> 4]);
		putchar(hex[digest[j] & 0x0f]);
	    }
	    putchar('\n');
	    free(digest);
	}
    }

    return 0;
}
