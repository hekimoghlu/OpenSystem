/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 1, 2025.
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
#include <fcntl.h>
#include <unistd.h>
#if defined(HAVE_STDINT_H)
# include <stdint.h>
#elif defined(HAVE_INTTYPES_H)
# include <inttypes.h>
#endif

#include "sudoers.h"

static int fuzz_printf(int msg_type, const char *fmt, ...);
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

/* Required to link with parser. */
struct sudo_user sudo_user;
struct passwd *list_pw;
sudo_printf_t sudo_printf = fuzz_printf;

FILE *
open_sudoers(const char *file, bool doedit, bool *keepopen)
{
    /*
     * If we allow the fuzzer to choose include paths it will
     * include random files in the file system.
     * This leads to bug reports that cannot be reproduced.
     */
    return NULL;
}

static int
fuzz_printf(int msg_type, const char *fmt, ...)
{
    return 0;
}

bool
init_envtables(void)
{
    return true;
}

int
set_cmnd_path(const char *runchroot)
{
    /* Cannot return FOUND without also setting user_cmnd to a new value. */
    return NOT_FOUND;
}

static FILE *
open_data(const uint8_t *data, size_t size)
{
#ifdef HAVE_FMEMOPEN
    /* Operate in-memory. */
    return fmemopen((void *)data, size, "r");
#else
    char tempfile[] = "/tmp/ldif.XXXXXX";
    size_t nwritten;
    int fd;

    /* Use (unlinked) temporary file. */
    fd = mkstemp(tempfile);
    if (fd == -1)
	return NULL;
    unlink(tempfile);
    nwritten = write(fd, data, size);
    if (nwritten != size) {
	close(fd);
	return NULL;
    }
    lseek(fd, 0, SEEK_SET);
    return fdopen(fd, "r");
#endif
}

static int
fuzz_conversation(int num_msgs, const struct sudo_conv_message msgs[],
    struct sudo_conv_reply replies[], struct sudo_conv_callback *callback)
{
    int n;

    for (n = 0; n < num_msgs; n++) {
	const struct sudo_conv_message *msg = &msgs[n];

	switch (msg->msg_type & 0xff) {
	    case SUDO_CONV_PROMPT_ECHO_ON:
	    case SUDO_CONV_PROMPT_MASK:
	    case SUDO_CONV_PROMPT_ECHO_OFF:
		/* input not supported */
		return -1;
	    case SUDO_CONV_ERROR_MSG:
	    case SUDO_CONV_INFO_MSG:
		/* no output for fuzzers */
		break;
	    default:
		return -1;
	}
    }
    return 0;
}

int
LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    struct sudoers_parse_tree parse_tree;
    FILE *fp;

    /* Don't waste time fuzzing tiny inputs. */
    if (size < 5)
        return 0;

    fp = open_data(data, size);
    if (fp == NULL)
        return 0;

    initprogname("fuzz_sudoers_ldif");
    sudoers_debug_register(getprogname(), NULL);
    if (getenv("SUDO_FUZZ_VERBOSE") == NULL)
	sudo_warn_set_conversation(fuzz_conversation);

    /* Initialize defaults and parse LDIF-format sudoers. */
    init_defaults();
    init_parse_tree(&parse_tree, NULL, NULL);
    sudoers_parse_ldif(&parse_tree, fp, "ou=SUDOers,dc=sudo,dc=ws", true);

    /* Cleanup. */
    free_parse_tree(&parse_tree);
    fclose(fp);
    sudoers_debug_deregister();
    fflush(stdout);

    return 0;
}
