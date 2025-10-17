/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 14, 2022.
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
/* System library. */

#include <sys_defs.h>

/* Utility library. */

#include <vstream.h>
#include <argv.h>
#include <dict.h>
#include <msg.h>

/* Global library. */

#include <mbox_conf.h>

/* XSASL library. */

#include <xsasl.h>

/* TLS library. */

#include <tls.h>

/* Application-specific. */

#include <postconf.h>

/* pcf_show_maps - show available maps */

void    pcf_show_maps(void)
{
    ARGV   *maps_argv;
    int     i;

    maps_argv = dict_mapnames();
    for (i = 0; i < maps_argv->argc; i++)
	vstream_printf("%s\n", maps_argv->argv[i]);
    argv_free(maps_argv);
}

/* pcf_show_locks - show available mailbox locking methods */

void    pcf_show_locks(void)
{
    ARGV   *locks_argv;
    int     i;

    locks_argv = mbox_lock_names();
    for (i = 0; i < locks_argv->argc; i++)
	vstream_printf("%s\n", locks_argv->argv[i]);
    argv_free(locks_argv);
}

/* pcf_show_sasl - show SASL plug-in types */

void    pcf_show_sasl(int what)
{
    ARGV   *sasl_argv;
    int     i;

    sasl_argv = (what & PCF_SHOW_SASL_SERV) ? xsasl_server_types() :
	xsasl_client_types();
    for (i = 0; i < sasl_argv->argc; i++)
	vstream_printf("%s\n", sasl_argv->argv[i]);
    argv_free(sasl_argv);
}

/* pcf_show_tls - show TLS support */

void    pcf_show_tls(const char *what)
{
#ifdef USE_TLS
    if (strcmp(what, "compile-version") == 0)
	vstream_printf("%s\n", tls_compile_version());
    else if (strcmp(what, "run-version") == 0)
	vstream_printf("%s\n", tls_run_version());
    else if (strcmp(what, "public-key-algorithms") == 0) {
	const char **cpp;

	for (cpp = tls_pkey_algorithms(); *cpp; cpp++)
	    vstream_printf("%s\n", *cpp);
    } else {
	msg_warn("unknown 'postconf -T' mode: %s", what);
	exit(1);
    }
#endif						/* USE_TLS */
}
