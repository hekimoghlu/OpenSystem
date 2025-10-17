/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 26, 2023.
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
 * Copyright (c) 2003-2004 Networks Associates Technology, Inc.
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

#include <sys/cdefs.h>

#include <sys/param.h>
#include <sys/kernel.h>
#include <sys/lock.h>
#include <sys/malloc.h>
#include <sys/sbuf.h>
#include <sys/systm.h>
#include <sys/vnode.h>
#include <sys/mount.h>
#include <sys/file.h>
#include <sys/namei.h>
#include <sys/sysctl.h>
#include <sys/shm.h>
#include <sys/shm_internal.h>

#include <security/mac_internal.h>


void
mac_sysvshm_label_init(struct shmid_kernel *shmsegptr)
{
	mac_labelzone_alloc_owned(&shmsegptr->label, MAC_WAITOK, ^(struct label *label) {
		MAC_PERFORM(sysvshm_label_init, label);
	});
}

struct label *
mac_sysvshm_label(struct shmid_kernel *shmsegptr)
{
	return mac_label_verify(&shmsegptr->label);
}

void
mac_sysvshm_label_destroy(struct shmid_kernel *shmsegptr)
{
	mac_labelzone_free_owned(&shmsegptr->label, ^(struct label *label) {
		MAC_PERFORM(sysvshm_label_destroy, label);
	});
}

void
mac_sysvshm_label_associate(struct ucred *cred, struct shmid_kernel *shmsegptr)
{
	MAC_PERFORM(sysvshm_label_associate, cred, shmsegptr, mac_sysvshm_label(shmsegptr));
}

void
mac_sysvshm_label_recycle(struct shmid_kernel *shmsegptr)
{
	MAC_PERFORM(sysvshm_label_recycle, mac_sysvshm_label(shmsegptr));
}

int
mac_sysvshm_check_shmat(struct ucred *cred, struct shmid_kernel *shmsegptr,
    int shmflg)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_sysvshm_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(sysvshm_check_shmat, cred, shmsegptr, mac_sysvshm_label(shmsegptr),
	    shmflg);

	return error;
}

int
mac_sysvshm_check_shmctl(struct ucred *cred, struct shmid_kernel *shmsegptr,
    int cmd)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_sysvshm_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(sysvshm_check_shmctl, cred, shmsegptr, mac_sysvshm_label(shmsegptr),
	    cmd);

	return error;
}

int
mac_sysvshm_check_shmdt(struct ucred *cred, struct shmid_kernel *shmsegptr)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_sysvshm_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(sysvshm_check_shmdt, cred, shmsegptr, mac_sysvshm_label(shmsegptr));

	return error;
}

int
mac_sysvshm_check_shmget(struct ucred *cred, struct shmid_kernel *shmsegptr,
    int shmflg)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_sysvshm_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(sysvshm_check_shmget, cred, shmsegptr, mac_sysvshm_label(shmsegptr),
	    shmflg);

	return error;
}
