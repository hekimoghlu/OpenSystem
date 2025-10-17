/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 16, 2022.
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
#if __has_include(<CrashReporterClient.h>)
#include <CrashReporterClient.h>
#else
#define CRSetCrashLogMessage(...)
#endif
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <signal.h>
#include <sys/signal.h>
#include <errno.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wstrict-prototypes"

extern int __platform_sigaction (int sig,
		const struct sigaction * __restrict nsv,
		struct sigaction * __restrict osv);

int
sigaction (int sig, const struct sigaction * __restrict nsv,
		struct sigaction * __restrict osv)
{
	int ret = __platform_sigaction(sig, nsv, osv);
#ifdef FEATURE_SIGNAL_RESTRICTION
	// Note: The "sig != 0" here is to force the compiler to maintain that "sig"
	// is live, and in a register, after __sigaction so it is visible in the
	// crashing register state.
	if (ret == -1 && errno == ENOTSUP && sig != 0) {
		CRSetCrashLogMessage("sigaction on fatal signals is not supported");
		__builtin_trap();
	}
#endif
	return ret;
}

// XXX
#ifdef __DYNAMIC__

int
_sigaction_nobind (sig, nsv, osv)
        int sig;
	register const struct sigaction *nsv;
        register struct sigaction *osv;
{
    return sigaction(sig, nsv, osv);
}
#endif

#pragma clang diagnostic pop
