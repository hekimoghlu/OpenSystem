/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 14, 2024.
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

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <regex.h>
#include <time.h>
#include <unistd.h>
#if defined(HAVE_STDINT_H)
# include <stdint.h>
#elif defined(HAVE_INTTYPES_H)
# include <inttypes.h>
#endif

#include "sudo_compat.h"
#include "sudo_conf.h"
#include "sudo_debug.h"
#include "sudo_eventlog.h"
#include "sudo_fatal.h"
#include "sudo_iolog.h"
#include "sudo_plugin.h"
#include "sudo_util.h"

#include "logsrvd.h"

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

/*
 * Stub version that always succeeds for small inputs and fails for large.
 * We want to fuzz our parser, not libc's regular expression code.
 */
bool
sudo_regex_compile_v1(void *v, const char *pattern, const char **errstr)
{
    regex_t *preg = v;

    if (strlen(pattern) > 32) {
	*errstr = "invalid regular expression";
	return false;
    }

    /* hopefully avoid regfree() crashes */
    memset(preg, 0, sizeof(*preg));
    return true;
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
    char tempfile[] = "/tmp/logsrvd_conf.XXXXXX";
    size_t nwritten;
    int fd;

    initprogname("fuzz_logsrvd_conf");
    if (getenv("SUDO_FUZZ_VERBOSE") == NULL)
	sudo_warn_set_conversation(fuzz_conversation);

    /* logsrvd_conf_read() uses a conf file path, not an open file. */
    fd = mkstemp(tempfile);
    if (fd == -1)
	return 0;
    nwritten = write(fd, data, size);
    if (nwritten != size) {
	close(fd);
	return 0;
    }
    close(fd);

    if (logsrvd_conf_read(tempfile)) {
	/* public config getters */
	logsrvd_conf_iolog_dir();
	logsrvd_conf_iolog_file();
	logsrvd_conf_iolog_mode();
	logsrvd_conf_pid_file();
	logsrvd_conf_relay_address();
	logsrvd_conf_relay_connect_timeout();
	logsrvd_conf_relay_tcp_keepalive();
	logsrvd_conf_relay_timeout();
	logsrvd_conf_server_listen_address();
	logsrvd_conf_server_tcp_keepalive();
	logsrvd_conf_server_timeout();

	/* free config */
	logsrvd_conf_cleanup();
    }

    unlink(tempfile);

    fflush(stdout);

    return 0;
}
