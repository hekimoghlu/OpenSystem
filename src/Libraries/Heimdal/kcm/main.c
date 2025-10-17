/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 2, 2022.
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
#include "kcm_locl.h"

#ifdef __APPLE__
#include <sandbox.h>
#include <os/log.h>
#include <os/log_private.h>
#endif

sig_atomic_t exit_flag = 0;

krb5_context kcm_context = NULL;

const char *service_name = "org.h5l.kcm";

static void
terminated(void *ctx)
{
    exit(0);
}

static void
sigusr1(void *ctx)
{
    kcm_debug_ccache(kcm_context);
}

static void
sigusr2(void *ctx)
{
    kcm_debug_events(kcm_context);
}

static void
timeout_handler(void)
{
    kcm_write_dump(kcm_context);
    exit(0);
}

int
main(int argc, char **argv)
{
    krb5_error_code ret;
    setprogname(argv[0]);

#ifdef __APPLE__
    /* Tell logd we're special */
    os_log_set_client_type(OS_LOG_CLIENT_TYPE_LOGD_DEPENDENCY, 0);
#endif

    ret = krb5_init_context(&kcm_context);
    if (ret) {
	errx (1, "krb5_init_context failed: %d", ret);
	return ret;
    }

    kcm_configure(argc, argv);

#ifdef HAVE_SIGACTION
    {
	struct sigaction sa;

	sa.sa_flags = 0;
	sa.sa_handler = SIG_IGN;
	sigemptyset(&sa.sa_mask);
	sigaction(SIGPIPE, &sa, NULL);
    }
#else
    signal(SIGPIPE, SIG_IGN);
#endif

    heim_sipc_signal_handler(SIGINT, terminated, "SIGINT");
    heim_sipc_signal_handler(SIGTERM, terminated, "SIGTERM");
    heim_sipc_signal_handler(SIGUSR1, sigusr1, NULL);
    heim_sipc_signal_handler(SIGUSR2, sigusr2, NULL);
#ifdef SIGXCPU
    heim_sipc_signal_handler(SIGXCPU, terminated, "CPU time limit exceeded");
#endif


#ifdef SUPPORT_DETACH
    if (detach_from_console)
	daemon(0, 0);
#endif
    kcm_session_setup_handler();

    kcm_read_dump(kcm_context);
    
    if (launchd_flag) {
	heim_sipc mach;
	heim_sipc_launchd_mach_init(service_name, kcm_service, NULL, &mach);
    } else {
	heim_sipc un;
	heim_sipc_service_unix(service_name, kcm_service, NULL, &un);
    }

#ifdef __APPLE__
    {
	char *errorstring;
	ret = sandbox_init("kcm", SANDBOX_NAMED, &errorstring);
	if (ret)
	    errx(1, "sandbox_init failed: %d: %s", ret, errorstring);
    }
#endif /* __APPLE__ */

    heim_sipc_set_timeout_handler(timeout_handler);

    /*
     * If we have didn't get a time or, and it was non zero,
     * lets set the timeout
     */
    if (kcm_timeout < 0)
	kcm_timeout = 15;
    if (kcm_timeout != 0)
	heim_sipc_timeout(kcm_timeout);

    heim_ipc_main();

    krb5_free_context(kcm_context);
    return 0;
}
