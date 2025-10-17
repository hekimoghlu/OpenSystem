/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 19, 2021.
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
/*-
 * Copyright (c) 2006 nCircle Network Security, Inc.
 * Copyright (c) 2009 Robert N. M. Watson
 * All rights reserved.
 *
 * This software was developed by Robert N. M. Watson for the TrustedBSD
 * Project under contract to nCircle Network Security, Inc.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR, NCIRCLE NETWORK SECURITY,
 * INC., OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <sys/cdefs.h>

#include <sys/param.h>
#include <sys/kernel.h>
#include <sys/priv.h>
#include <sys/proc.h>
#include <sys/systm.h>
#include <sys/kauth.h>

#if CONFIG_MACF
#include <security/mac_framework.h>
#endif

int proc_check_footprint_priv(void);

/*
 * Check a credential for privilege.  Lots of good reasons to deny privilege;
 * only a few to grant it.
 */
int
priv_check_cred(kauth_cred_t cred, int priv, int flags)
{
#if !CONFIG_MACF
#pragma unused(priv)
#endif
	int error;

	/*
	 * We first evaluate policies that may deny the granting of
	 * privilege unilaterally.
	 */
#if CONFIG_MACF
	error = mac_priv_check(cred, priv);
	if (error) {
		goto out;
	}
#endif

	/* Only grant all privileges to root if DEFAULT_UNPRIVELEGED flag is NOT set. */
	if (!(flags & PRIVCHECK_DEFAULT_UNPRIVILEGED_FLAG)) {
		/*
		 * Having determined if privilege is restricted by various policies,
		 * now determine if privilege is granted.	At this point, any policy
		 * may grant privilege.	For now, we allow short-circuit boolean
		 * evaluation, so may not call all policies.	 Perhaps we should.
		 */
		if (kauth_cred_getuid(cred) == 0) {
			error = 0;
			goto out;
		}
	}

	/*
	 * Now check with MAC, if enabled, to see if a policy module grants
	 * privilege.
	 */
#if CONFIG_MACF
	if (mac_priv_grant(cred, priv) == 0) {
		error = 0;
		goto out;
	}
#endif

	/*
	 * The default is deny, so if no policies have granted it, reject
	 * with a privilege error here.
	 */
	error = EPERM;
out:
	return error;
}

int
proc_check_footprint_priv(void)
{
	return priv_check_cred(kauth_cred_get(), PRIV_VM_FOOTPRINT_LIMIT, 0);
}
