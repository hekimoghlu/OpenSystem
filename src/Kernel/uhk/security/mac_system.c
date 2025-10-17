/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 15, 2022.
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
 * Copyright (c) 1999, 2000, 2001, 2002 Robert N. M. Watson
 * Copyright (c) 2001 Ilmar S. Habibulin
 * Copyright (c) 2001, 2002, 2003, 2004 Networks Associates Technology, Inc.
 *
 * This software was developed by Robert Watson and Ilmar Habibulin for the
 * TrustedBSD Project.
 *
 * This software was developed for the FreeBSD Project in part by Network
 * Associates Laboratories, the Security Research Division of Network
 * Associates, Inc. under DARPA/SPAWAR contract N66001-01-C-8035 ("CBOSS"),
 * as part of the DARPA CHATS research program.
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
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 */

#include <sys/param.h>
#include <sys/vnode.h>
#include <sys/vnode_internal.h>

#include <security/mac_internal.h>


int
mac_system_check_acct(kauth_cred_t cred, struct vnode *vp)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_system_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(system_check_acct, cred, vp,
	    vp != NULL ? mac_vnode_label(vp) : NULL);

	return error;
}

int
mac_system_check_host_priv(kauth_cred_t cred)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_system_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(system_check_host_priv, cred);

	return error;
}

int
mac_system_check_info(kauth_cred_t cred, const char *info_type)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_system_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(system_check_info, cred, info_type);

	return error;
}

int
mac_system_check_nfsd(kauth_cred_t cred)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_system_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(system_check_nfsd, cred);

	return error;
}

int
mac_system_check_reboot(kauth_cred_t cred, int howto)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_system_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(system_check_reboot, cred, howto);

	return error;
}


int
mac_system_check_settime(kauth_cred_t cred)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_system_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(system_check_settime, cred);

	return error;
}

int
mac_system_check_swapon(kauth_cred_t cred, struct vnode *vp)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_system_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(system_check_swapon, cred, vp, mac_vnode_label(vp));
	return error;
}

int
mac_system_check_swapoff(kauth_cred_t cred, struct vnode *vp)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_system_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(system_check_swapoff, cred, vp, mac_vnode_label(vp));
	return error;
}

int
mac_system_check_sysctlbyname(kauth_cred_t cred, const char *namestring, int *name,
    size_t namelen, user_addr_t oldctl, size_t oldlen,
    user_addr_t newctl, size_t newlen)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_system_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(system_check_sysctlbyname, cred, namestring,
	    name, namelen, oldctl, oldlen, newctl, newlen);

	return error;
}

int
mac_system_check_kas_info(kauth_cred_t cred, int selector)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_system_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(system_check_kas_info, cred, selector);

	return error;
}
