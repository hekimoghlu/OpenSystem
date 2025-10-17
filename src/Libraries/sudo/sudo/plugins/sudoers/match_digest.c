/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 1, 2022.
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
/*
 * This is an open source non-commercial project. Dear PVS-Studio, please check it.
 * PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
 */

#include <config.h>

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "sudoers.h"
#include "sudo_digest.h"
#include <gram.h>

bool
digest_matches(int fd, const char *path, const char *runchroot,
    const struct command_digest_list *digests)
{
    unsigned int digest_type = SUDO_DIGEST_INVALID;
    unsigned char *file_digest = NULL;
    unsigned char *sudoers_digest = NULL;
    struct command_digest *digest;
    size_t digest_len = (size_t)-1;
    char pathbuf[PATH_MAX];
    bool matched = false;
    debug_decl(digest_matches, SUDOERS_DEBUG_MATCH);

    if (TAILQ_EMPTY(digests)) {
	/* No digest, no problem. */
	debug_return_bool(true);
    }

    if (fd == -1) {
	/* No file, no match. */
	goto done;
    }

    if (runchroot != NULL) {
	const int len =
	    snprintf(pathbuf, sizeof(pathbuf), "%s%s", runchroot, path);
	if (len >= ssizeof(pathbuf)) {
	    errno = ENAMETOOLONG;
	    debug_return_bool(false);
	}
	path = pathbuf;
    }

    TAILQ_FOREACH(digest, digests, entries) {
	/* Compute file digest if needed. */
	if (digest->digest_type != digest_type) {
	    free(file_digest);
	    file_digest = sudo_filedigest(fd, path, digest->digest_type,
		&digest_len);
	    if (lseek(fd, (off_t)0, SEEK_SET) == -1) {
		sudo_debug_printf(SUDO_DEBUG_ERROR|SUDO_DEBUG_ERRNO|SUDO_DEBUG_LINENO,
		    "unable to rewind digest fd");
	    }
	    digest_type = digest->digest_type;
	}
	if (file_digest == NULL) {
	    /* Warning (if any) printed by sudo_filedigest() */
	    goto done;
	}

	/* Convert the command digest from ascii to binary. */
	if ((sudoers_digest = malloc(digest_len)) == NULL) {
	    sudo_warnx(U_("%s: %s"), __func__, U_("unable to allocate memory"));
	    goto done;
	}
	if (strlen(digest->digest_str) == digest_len * 2) {
	    /* Convert ascii hex to binary. */
	    unsigned int i;
	    for (i = 0; i < digest_len; i++) {
		const int h = sudo_hexchar(&digest->digest_str[i + i]);
		if (h == -1)
		    goto bad_format;
		sudoers_digest[i] = (unsigned char)h;
	    }
	} else {
	    /* Convert base64 to binary. */
	    size_t len = base64_decode(digest->digest_str, sudoers_digest, digest_len);
	    if (len == (size_t)-1)
		goto bad_format;
	    if (len != digest_len) {
		sudo_warnx(
		    U_("digest for %s (%s) bad length %zu, expected %zu"),
		    path, digest->digest_str, len, digest_len);
		goto done;
	    }
	}
	if (memcmp(file_digest, sudoers_digest, digest_len) == 0) {
	    matched = true;
	    break;
	}

	sudo_debug_printf(SUDO_DEBUG_DIAG|SUDO_DEBUG_LINENO,
	    "%s digest mismatch for %s, expecting %s",
	    digest_type_to_name(digest->digest_type), path, digest->digest_str);
	free(sudoers_digest);
	sudoers_digest = NULL;
    }
    goto done;

bad_format:
    sudo_warnx(U_("digest for %s (%s) is not in %s form"), path,
	digest->digest_str, digest_type_to_name(digest->digest_type));
done:
    free(sudoers_digest);
    free(file_digest);
    debug_return_bool(matched);
}
