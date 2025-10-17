/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 16, 2022.
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
#include "kdc_locl.h"
#ifdef HAVE_UTIL_H
#include <util.h>
#endif

#ifdef __APPLE__
#include <sandbox.h>

int sandbox_flag = 1;
#endif

#ifdef HAVE_CAPNG
#include <cap-ng.h>
#endif

static void terminated(void *ctx) __attribute__((noreturn));


#ifdef SUPPORT_DETACH
int detach_from_console = -1;
#endif

/*
 * Allow dropping root bit, since heimdal reopens the database all the
 * time the database needs to be owned by the user you are switched
 * too. A better solution is to split the kdc in to more processes and
 * run the network facing part with very low privilege.
 */

static void
switch_environment(void)
{
#ifdef HAVE_GETEUID
    if ((runas_string || chroot_string) && geteuid() != 0)
	errx(1, "no running as root, can't switch user/chroot");

    if (chroot_string && chroot(chroot_string) != 0)
	errx(1, "chroot(%s)", "chroot_string failed");

    if (runas_string) {
	struct passwd *pw;

	pw = getpwnam(runas_string);
	if (pw == NULL)
	    errx(1, "unknown user %s", runas_string);

	if (initgroups(pw->pw_name, pw->pw_gid) < 0)
	    err(1, "initgroups failed");

#ifndef HAVE_CAPNG
	if (setgid(pw->pw_gid) < 0)
	    err(1, "setgid(%s) failed", runas_string);

	if (setuid(pw->pw_uid) < 0)
	    err(1, "setuid(%s)", runas_string);
#else
	capng_clear (CAPNG_EFFECTIVE | CAPNG_PERMITTED);
	if (capng_updatev (CAPNG_ADD, CAPNG_EFFECTIVE | CAPNG_PERMITTED,
	                   CAP_NET_BIND_SERVICE, CAP_SETPCAP, -1) < 0)
	    err(1, "capng_updateev");

	if (capng_change_id(pw->pw_uid, pw->pw_gid,
	                    CAPNG_CLEAR_BOUNDING) < 0)
	    err(1, "capng_change_id(%s)", runas_string);
#endif
    }
#endif
}

static krb5_context context;
static krb5_kdc_configuration *config;

static heim_array_t
get_realms(void)
{
    heim_array_t array;
    char **realms, **r;
    unsigned int i;
    int ret;

    array = heim_array_create();

    for(i = 0; i < config->num_db; i++) {

	if (config->db[i]->hdb_get_realms == NULL)
	    continue;
	
	ret = (config->db[i]->hdb_get_realms)(context, config->db[i], &realms);
	if (ret == 0) {
	    for (r = realms; r && *r; r++) {
		heim_string_t s = heim_string_create(*r);
		if (s)
		    heim_array_append_value(array, s);
		heim_release(s);
	    }
	    krb5_free_host_realm(context, realms);
	}
    }

    return array;
}

static void
terminated(void *ctx)
{
    kdc_log(context, config, 0, "Terminated: %s", (char *)ctx);
    exit(1);
}

int
main(int argc, char **argv)
{
    krb5_error_code ret;

    setprogname(argv[0]);

    ret = krb5_init_context(&context);
    if (ret == KRB5_CONFIG_BADFORMAT)
	errx (1, "krb5_init_context failed to parse configuration file");
    else if (ret)
	errx (1, "krb5_init_context failed: %d", ret);

    ret = krb5_kt_register(context, &hdb_kt_ops);
    if (ret)
	errx (1, "krb5_kt_register(HDB) failed: %d", ret);

    config = configure(context, argc, argv);

#ifdef SIGPIPE
#ifdef HAVE_SIGACTION
    {
	struct sigaction sa;

	sa.sa_flags = 0;
	sigemptyset(&sa.sa_mask);

	sa.sa_handler = SIG_IGN;
	sigaction(SIGPIPE, &sa, NULL);
    }
#else
    signal(SIGPIPE, SIG_IGN);
#endif
#endif /* SIGPIPE */



#ifdef SUPPORT_DETACH
    if (detach_from_console)
	daemon(0, 0);
#endif
#ifdef __APPLE__
    if (sandbox_flag) {
	char *errorstring;
	ret = sandbox_init("kdc", SANDBOX_NAMED, &errorstring);
	if (ret)
	    errx(1, "sandbox_init failed: %d: %s", ret, errorstring);
    }
    bonjour_announce(get_realms);
#endif /* __APPLE__ */
    pidfile(NULL);

    switch_environment();

    setup_listeners(context, config, listen_on_ipc, listen_on_network);

    heim_sipc_signal_handler(SIGINT, terminated, "SIGINT");
    heim_sipc_signal_handler(SIGTERM, terminated, "SIGTERM");
#ifdef SIGXCPU
    heim_sipc_signal_handler(SIGXCPU, terminated, "CPU time limit exceeded");
#endif

    heim_ipc_main();
}
