/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 27, 2022.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

#include "sudoers.h"
#include "sudo_digest.h"

unsigned char *
sudo_filedigest(int fd, const char *file, int digest_type, size_t *digest_len)
{
    unsigned char *file_digest = NULL;
    unsigned char buf[32 * 1024];
    struct sudo_digest *dig = NULL;
    FILE *fp = NULL;
    size_t nread;
    int fd2;
    debug_decl(sudo_filedigest, SUDOERS_DEBUG_UTIL);

    *digest_len = sudo_digest_getlen(digest_type);
    if (*digest_len == (size_t)-1) {
	sudo_warnx(U_("unsupported digest type %d for %s"), digest_type, file);
	goto bad;
    }

    if ((dig = sudo_digest_alloc(digest_type)) == NULL) {
	sudo_warnx(U_("%s: %s"), __func__, U_("unable to allocate memory"));
	goto bad;
    }

    if ((fd2 = dup(fd)) == -1) {
	sudo_debug_printf(SUDO_DEBUG_INFO, "unable to dup %s: %s",
	    file, strerror(errno));
	goto bad;
    }
    if ((fp = fdopen(fd2, "r")) == NULL) {
	sudo_debug_printf(SUDO_DEBUG_INFO, "unable to fdopen %s: %s",
	    file, strerror(errno));
	close(fd2);
	goto bad;
    }
    if ((file_digest = malloc(*digest_len)) == NULL) {
	sudo_warnx(U_("%s: %s"), __func__, U_("unable to allocate memory"));
	goto bad;
    }

    while ((nread = fread(buf, 1, sizeof(buf), fp)) != 0) {
	sudo_digest_update(dig, buf, nread);
    }
    if (ferror(fp)) {
	sudo_warnx(U_("%s: read error"), file);
	goto bad;
    }
    sudo_digest_final(dig, file_digest);
    sudo_digest_free(dig);
    fclose(fp);

    debug_return_ptr(file_digest);
bad:
    sudo_digest_free(dig);
    free(file_digest);
    if (fp != NULL)
	fclose(fp);
    debug_return_ptr(NULL);
}
