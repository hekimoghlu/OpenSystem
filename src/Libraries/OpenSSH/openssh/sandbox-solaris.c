/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 8, 2025.
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
#include "includes.h"

#ifdef SANDBOX_SOLARIS
#ifndef USE_SOLARIS_PRIVS
# error "--with-solaris-privs must be used with the Solaris sandbox"
#endif

#include <sys/types.h>

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#ifdef HAVE_PRIV_H
# include <priv.h>
#endif

#include "log.h"
#include "ssh-sandbox.h"
#include "xmalloc.h"

struct ssh_sandbox {
	priv_set_t *pset;
};

struct ssh_sandbox *
ssh_sandbox_init(struct monitor *monitor)
{
	struct ssh_sandbox *box = NULL;

	box = xcalloc(1, sizeof(*box));

	/* Start with "basic" and drop everything we don't need. */
	box->pset = solaris_basic_privset();

	if (box->pset == NULL) {
		free(box);
		return NULL;
	}

	/* Drop everything except the ability to use already-opened files */
	if (priv_delset(box->pset, PRIV_FILE_LINK_ANY) != 0 ||
#ifdef PRIV_NET_ACCESS
	    priv_delset(box->pset, PRIV_NET_ACCESS) != 0 ||
#endif
#ifdef PRIV_DAX_ACCESS
	    priv_delset(box->pset, PRIV_DAX_ACCESS) != 0 ||
#endif
#ifdef PRIV_SYS_IB_INFO
	    priv_delset(box->pset, PRIV_SYS_IB_INFO) != 0 ||
#endif
	    priv_delset(box->pset, PRIV_PROC_EXEC) != 0 ||
	    priv_delset(box->pset, PRIV_PROC_FORK) != 0 ||
	    priv_delset(box->pset, PRIV_PROC_INFO) != 0 ||
	    priv_delset(box->pset, PRIV_PROC_SESSION) != 0) {
		free(box);
		return NULL;
	}

	/* These may not be available on older Solaris-es */
# if defined(PRIV_FILE_READ) && defined(PRIV_FILE_WRITE)
	if (priv_delset(box->pset, PRIV_FILE_READ) != 0 ||
	    priv_delset(box->pset, PRIV_FILE_WRITE) != 0) {
		free(box);
		return NULL;
	}
# endif

	return box;
}

void
ssh_sandbox_child(struct ssh_sandbox *box)
{
	if (setppriv(PRIV_SET, PRIV_PERMITTED, box->pset) != 0 ||
	    setppriv(PRIV_SET, PRIV_LIMIT, box->pset) != 0 ||
	    setppriv(PRIV_SET, PRIV_INHERITABLE, box->pset) != 0)
		fatal("setppriv: %s", strerror(errno));
}

void
ssh_sandbox_parent_finish(struct ssh_sandbox *box)
{
	priv_freeset(box->pset);
	box->pset = NULL;
	free(box);
}

void
ssh_sandbox_parent_preauth(struct ssh_sandbox *box, pid_t child_pid)
{
	/* Nothing to do here */
}

#endif /* SANDBOX_SOLARIS */
