/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 29, 2022.
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
 * Copyright (c) 2002, 2003 Networks Associates Technology, Inc.
 * Copyright (c) 2005 SPARTA, Inc.
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
#include <sys/vnode.h>
#include <sys/pipe.h>
#include <sys/sysctl.h>

#include <security/mac_internal.h>


static struct label *
mac_pipe_label_alloc(void)
{
	struct label *label;

	label = mac_labelzone_alloc(MAC_WAITOK);
	if (label == NULL) {
		return NULL;
	}
	MAC_PERFORM(pipe_label_init, label);
	return label;
}

struct label *
mac_pipe_label(struct pipe *cpipe)
{
	return cpipe->pipe_label;
}

void
mac_pipe_set_label(struct pipe *cpipe, struct label *label)
{
	cpipe->pipe_label = label;
}

void
mac_pipe_label_init(struct pipe *cpipe)
{
	mac_pipe_set_label(cpipe, mac_pipe_label_alloc());
}

void
mac_pipe_label_free(struct label *label)
{
	MAC_PERFORM(pipe_label_destroy, label);
	mac_labelzone_free(label);
}

void
mac_pipe_label_destroy(struct pipe *cpipe)
{
	struct label *label = mac_pipe_label(cpipe);
	mac_pipe_set_label(cpipe, NULL);
	mac_pipe_label_free(label);
}

void
mac_pipe_label_associate(kauth_cred_t cred, struct pipe *cpipe)
{
	MAC_PERFORM(pipe_label_associate, cred, cpipe, mac_pipe_label(cpipe));
}

int
mac_pipe_check_kqfilter(kauth_cred_t cred, struct knote *kn,
    struct pipe *cpipe)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_pipe_enforce) {
		return 0;
	}
#endif
	MAC_CHECK(pipe_check_kqfilter, cred, kn, cpipe, mac_pipe_label(cpipe));
	return error;
}
int
mac_pipe_check_ioctl(kauth_cred_t cred, struct pipe *cpipe, u_long cmd)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_pipe_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(pipe_check_ioctl, cred, cpipe, mac_pipe_label(cpipe), cmd);

	return error;
}

int
mac_pipe_check_read(kauth_cred_t cred, struct pipe *cpipe)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_pipe_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(pipe_check_read, cred, cpipe, mac_pipe_label(cpipe));

	return error;
}

int
mac_pipe_check_select(kauth_cred_t cred, struct pipe *cpipe, int which)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_pipe_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(pipe_check_select, cred, cpipe, mac_pipe_label(cpipe), which);

	return error;
}

int
mac_pipe_check_stat(kauth_cred_t cred, struct pipe *cpipe)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_pipe_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(pipe_check_stat, cred, cpipe, mac_pipe_label(cpipe));

	return error;
}

int
mac_pipe_check_write(kauth_cred_t cred, struct pipe *cpipe)
{
	int error;

#if SECURITY_MAC_CHECK_ENFORCE
	/* 21167099 - only check if we allow write */
	if (!mac_pipe_enforce) {
		return 0;
	}
#endif

	MAC_CHECK(pipe_check_write, cred, cpipe, mac_pipe_label(cpipe));

	return error;
}
