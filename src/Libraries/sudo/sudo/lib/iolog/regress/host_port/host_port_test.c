/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 29, 2021.
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
#ifdef HAVE_STDBOOL_H
# include <stdbool.h>
#else
# include "compat/stdbool.h"
#endif
#include <time.h>
#include <unistd.h>

#include "sudo_compat.h"
#include "sudo_fatal.h"
#include "sudo_iolog.h"
#include "sudo_util.h"

sudo_dso_public int main(int argc, char *argv[]);

/*
 * Test that iolog_parse_host_port() works as expected.
 */

struct host_port_test {
    const char *str;		/* input string */
    const char *host;		/* parsed host */
    const char *port;		/* parsed port */
    bool tls;			/* parsed TLS flag */
    const char *defport;	/* default port */
    const char *defport_tls;	/* default port */
    bool ret;			/* return value */
};

static struct host_port_test test_data[] = {
    /* No TLS */
    { "xerxes", "xerxes", "12345", false, "12345", NULL, true },
    { "xerxes:12345", "xerxes", "12345", false, "67890", NULL, true },
    { "127.0.0.1", "127.0.0.1", "12345", false, "12345", NULL, true },
    { "127.0.0.1:12345", "127.0.0.1", "12345", false, "67890", NULL, true },
    { "[::1]", "::1", "12345", false, "12345", NULL, true },
    { "[::1]:12345", "::1", "12345", false, "67890", NULL, true },

    /* With TLS */
    { "xerxes(tls)", "xerxes", "12345", true, "5678", "12345", true },
    { "xerxes:12345(tls)", "xerxes", "12345", true, "5678", "67890", true },
    { "127.0.0.1(tls)", "127.0.0.1", "12345", true, "5678", "12345", true },
    { "127.0.0.1:12345(tls)", "127.0.0.1", "12345", true, "5678", "67890", true },
    { "[::1](tls)", "::1", "12345", true, "5678", "12345", true },
    { "[::1]:12345(tls)", "::1", "12345", true, "5678", "67890", true },

    /* Errors */
    { "xerxes:", NULL, NULL, false, "12345", NULL, false },	/* missing port */
    { "127.0.0.1:", NULL, NULL, false, "12345", NULL, false },	/* missing port */
    { "[::1:12345", NULL, NULL, false, "67890", NULL, false },	/* missing bracket */
    { "[::1]:", NULL, NULL, false, "12345", NULL, false },	/* missing port */
    { NULL }
};

int
main(int argc, char *argv[])
{
    int i, errors = 0, ntests = 0;
    char *host, *port, *copy = NULL;
    bool ret, tls;

    initprogname(argc > 0 ? argv[0] : "host_port_test");

    for (i = 0; test_data[i].str != NULL; i++) {
	host = port = NULL;
	tls = false;
	free(copy);
	if ((copy = strdup(test_data[i].str)) == NULL)
	    sudo_fatal_nodebug(NULL);

	ntests++;
	ret = iolog_parse_host_port(copy, &host, &port, &tls,
	    test_data[i].defport, test_data[i].defport_tls);
	if (ret != test_data[i].ret) {
	    sudo_warnx_nodebug("test #%d: %s: returned %s, expected %s",
		ntests, test_data[i].str, ret ? "true" : "false",
		test_data[i].ret ? "true" : "false");
	    errors++;
	    continue;
	}
	if (!ret)
	    continue;

	if (host == NULL) {
	    sudo_warnx_nodebug("test #%d: %s: NULL host",
		ntests, test_data[i].str);
	    errors++;
	    continue;
	}
	if (strcmp(host, test_data[i].host) != 0) {
	    sudo_warnx_nodebug("test #%d: %s: bad host, expected %s, got %s",
		ntests, test_data[i].str, test_data[i].host, host);
	    errors++;
	    continue;
	}
	if (port == NULL) {
	    sudo_warnx_nodebug("test #%d: %s: NULL port",
		ntests, test_data[i].str);
	    errors++;
	    continue;
	}
	if (strcmp(port, test_data[i].port) != 0) {
	    sudo_warnx_nodebug("test #%d: %s: bad port, expected %s, got %s",
		ntests, test_data[i].str, test_data[i].port, port);
	    errors++;
	    continue;
	}
	if (tls != test_data[i].tls) {
	    sudo_warnx_nodebug("test #%d: %s: bad tls, expected %s, got %s",
		ntests, test_data[i].str, test_data[i].tls ? "true" : "false",
		tls ? "true" : "false");
	    errors++;
	    continue;
	}
    }
    free(copy);
    if (ntests != 0) {
	printf("%s: %d tests run, %d errors, %d%% success rate\n",
	    getprogname(), ntests, errors, (ntests - errors) * 100 / ntests);
    }
    return errors;
}
