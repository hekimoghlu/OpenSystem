/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 12, 2024.
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
#include <sys/systm.h>
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
#include <sys/msg.h>

#include <security/mac_internal.h>

void
mac_sysvmsg_label_init(struct msg *msgptr)
{
	mac_labelzone_alloc_owned(&msgptr->label, MAC_WAITOK, ^(struct label *label) {
		MAC_PERFORM(sysvmsg_label_init, label);
	});
}

struct label *
mac_sysvmsg_label(struct msg *msgptr)
{
	return mac_label_verify(&msgptr->label);
}

void
mac_sysvmsq_label_init(struct msqid_kernel *msqptr)
{
	mac_labelzone_alloc_owned(&msqptr->label, MAC_WAITOK, ^(struct label *label) {
		MAC_PERFORM(sysvmsq_label_init, label);
	});
}

struct label *
mac_sysvmsq_label(struct msqid_kernel *msqptr)
{
	return mac_label_verify(&msqptr->label);
}

void
mac_sysvmsg_label_associate(kauth_cred_t cred, struct msqid_kernel *msqptr,
    struct msg *msgptr)
{
	MAC_PERFORM(sysvmsg_label_associate, cred, msqptr, mac_sysvmsq_label(msqptr),
	    msgptr, mac_sysvmsg_label(msgptr));
}

void
mac_sysvmsq_label_associate(kauth_cred_t cred, struct msqid_kernel *msqptr)
{
	MAC_PERFORM(sysvmsq_label_associate, cred, msqptr, mac_sysvmsq_label(msqptr));
}

void
mac_sysvmsg_label_recycle(struct msg *msgptr)
{
	MAC_PERFORM(sysvmsg_label_recycle, mac_sysvmsg_label(msgptr));
}

void
mac_sysvmsq_label_recycle(struct msqid_kernel *msqptr)
{
	MAC_PERFORM(sysvmsq_label_recycle, mac_sysvmsq_label(msqptr));
}

int
mac_sysvmsq_check_enqueue(kauth_cred_t cred, struct msg *msgptr,
    struct msqid_kernel *msqptr)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_sysvmsg_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(sysvmsq_check_enqueue, cred, msgptr, mac_sysvmsg_label(msgptr), msqptr,
	    mac_sysvmsq_label(msqptr));

	return error;
}

int
mac_sysvmsq_check_msgrcv(kauth_cred_t cred, struct msg *msgptr)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_sysvmsg_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(sysvmsq_check_msgrcv, cred, msgptr, mac_sysvmsg_label(msgptr));

	return error;
}

int
mac_sysvmsq_check_msgrmid(kauth_cred_t cred, struct msg *msgptr)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_sysvmsg_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(sysvmsq_check_msgrmid, cred, msgptr, mac_sysvmsg_label(msgptr));

	return error;
}

int
mac_sysvmsq_check_msqget(kauth_cred_t cred, struct msqid_kernel *msqptr)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_sysvmsg_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(sysvmsq_check_msqget, cred, msqptr, mac_sysvmsq_label(msqptr));

	return error;
}

int
mac_sysvmsq_check_msqsnd(kauth_cred_t cred, struct msqid_kernel *msqptr)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_sysvmsg_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(sysvmsq_check_msqsnd, cred, msqptr, mac_sysvmsq_label(msqptr));

	return error;
}

int
mac_sysvmsq_check_msqrcv(kauth_cred_t cred, struct msqid_kernel *msqptr)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_sysvmsg_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(sysvmsq_check_msqrcv, cred, msqptr, mac_sysvmsq_label(msqptr));

	return error;
}

int
mac_sysvmsq_check_msqctl(kauth_cred_t cred, struct msqid_kernel *msqptr,
    int cmd)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_sysvmsg_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(sysvmsq_check_msqctl, cred, msqptr, mac_sysvmsq_label(msqptr), cmd);

	return error;
}
