/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 9, 2021.
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
#include <sys/posix_shm.h>
#include <sys/sbuf.h>
#include <sys/systm.h>
#include <sys/sysctl.h>
#include <sys/vnode.h>

#include <security/mac_internal.h>

void
mac_posixshm_label_init(struct pshminfo *pshm)
{
	mac_labelzone_alloc_owned(&pshm->pshm_label, MAC_WAITOK, ^(struct label *label) {
		MAC_PERFORM(posixshm_label_init, label);
	});
}

void
mac_posixshm_label_destroy(struct pshminfo *pshm)
{
	mac_labelzone_free_owned(&pshm->pshm_label, ^(struct label *label) {
		MAC_PERFORM(posixshm_label_destroy, label);
	});
}

struct label *
mac_posixshm_label(struct pshminfo *pshm)
{
	return mac_label_verify(&pshm->pshm_label);
}

void
mac_posixshm_vnode_label_associate(kauth_cred_t cred,
    struct pshminfo *pshm, struct label *plabel,
    vnode_t vp, struct label *vlabel)
{
	MAC_PERFORM(vnode_label_associate_posixshm, cred,
	    pshm, plabel, vp, vlabel);
}

void
mac_posixshm_label_associate(kauth_cred_t cred, struct pshminfo *pshm,
    const char *name)
{
	MAC_PERFORM(posixshm_label_associate, cred, pshm, mac_posixshm_label(pshm), name);
}

int
mac_posixshm_check_create(kauth_cred_t cred, const char *name)
{
	int error = 0;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_posixshm_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(posixshm_check_create, cred, name);

	return error;
}

int
mac_posixshm_check_open(kauth_cred_t cred, struct pshminfo *shm, int fflags)
{
	int error = 0;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_posixshm_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(posixshm_check_open, cred, shm, mac_posixshm_label(shm), fflags);

	return error;
}

int
mac_posixshm_check_mmap(kauth_cred_t cred, struct pshminfo *shm,
    int prot, int flags)
{
	int error = 0;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_posixshm_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(posixshm_check_mmap, cred, shm, mac_posixshm_label(shm),
	    prot, flags);

	return error;
}

int
mac_posixshm_check_stat(kauth_cred_t cred, struct pshminfo *shm)
{
	int error = 0;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_posixshm_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(posixshm_check_stat, cred, shm, mac_posixshm_label(shm));

	return error;
}

int
mac_posixshm_check_truncate(kauth_cred_t cred, struct pshminfo *shm,
    off_t size)
{
	int error = 0;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_posixshm_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(posixshm_check_truncate, cred, shm, mac_posixshm_label(shm), size);

	return error;
}

int
mac_posixshm_check_unlink(kauth_cred_t cred, struct pshminfo *shm,
    const char *name)
{
	int error = 0;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_posixshm_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(posixshm_check_unlink, cred, shm, mac_posixshm_label(shm), name);

	return error;
}
