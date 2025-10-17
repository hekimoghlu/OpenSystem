/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 7, 2024.
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
 * Copyright (c) 2003-2005 Networks Associates Technology, Inc.
 * All rights reserved.
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
 */

#include <sys/cdefs.h>

#include <sys/param.h>
#include <sys/kernel.h>
#include <sys/lock.h>
#include <sys/malloc.h>
#include <sys/sbuf.h>
#include <sys/systm.h>
#include <sys/sysctl.h>
#include <security/mac_internal.h>
#include <sys/posix_sem.h>


void
mac_posixsem_label_init(struct pseminfo *psem)
{
	mac_labelzone_alloc_owned(&psem->psem_label, MAC_WAITOK, ^(struct label *label) {
		MAC_PERFORM(posixsem_label_init, label);
	});
}

struct label *
mac_posixsem_label(struct pseminfo *psem)
{
	return mac_label_verify(&psem->psem_label);
}

void
mac_posixsem_label_destroy(struct pseminfo *psem)
{
	mac_labelzone_free_owned(&psem->psem_label, ^(struct label *label) {
		MAC_PERFORM(posixsem_label_destroy, label);
	});
}

void
mac_posixsem_label_associate(kauth_cred_t cred, struct pseminfo *psem,
    const char *name)
{
	MAC_PERFORM(posixsem_label_associate, cred, psem, mac_posixsem_label(psem), name);
}


void
mac_posixsem_vnode_label_associate(kauth_cred_t cred,
    struct pseminfo *psem, struct label *plabel,
    vnode_t vp, struct label *vlabel)
{
	MAC_PERFORM(vnode_label_associate_posixsem, cred,
	    psem, plabel, vp, vlabel);
}

int
mac_posixsem_check_create(kauth_cred_t cred, const char *name)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_posixsem_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(posixsem_check_create, cred, name);

	return error;
}

int
mac_posixsem_check_open(kauth_cred_t cred, struct pseminfo *psem)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_posixsem_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(posixsem_check_open, cred, psem,
	    mac_posixsem_label(psem));

	return error;
}

int
mac_posixsem_check_post(kauth_cred_t cred, struct pseminfo *psem)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_posixsem_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(posixsem_check_post, cred, psem, mac_posixsem_label(psem));

	return error;
}

int
mac_posixsem_check_unlink(kauth_cred_t cred, struct pseminfo *psem,
    const char *name)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_posixsem_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(posixsem_check_unlink, cred, psem, mac_posixsem_label(psem), name);

	return error;
}

int
mac_posixsem_check_wait(kauth_cred_t cred, struct pseminfo *psem)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_posixsem_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(posixsem_check_wait, cred, psem, mac_posixsem_label(psem));

	return error;
}
