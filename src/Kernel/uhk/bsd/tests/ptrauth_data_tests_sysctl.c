/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 10, 2021.
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
#if DEVELOPMENT || DEBUG
#if __has_feature(ptrauth_calls)

#include <sys/errno.h>
#include <sys/sysctl.h>
#include <sys/ubc.h>
#include <kern/kalloc.h>
#include <libkern/libkern.h>
#include <pexpert/pexpert.h>


#include <mach/task.h>
#include <kern/task.h>
#include <sys/ubc_internal.h>

extern kern_return_t ptrauth_data_tests(void);

/*
 * Given an existing PAC pointer (ptr), its declaration type (decl), the (key)
 * used to sign it and the string discriminator (discr), extract the raw pointer
 * along with the signature and compare it with one computed on the fly
 * via ptrauth_sign_unauthenticated().
 *
 * If the two mismatch, return an error and fail the test.
 */
#define VALIDATE_PTR(decl, ptr, key, discr) { \
	decl raw = *(decl *)(uintptr_t)&(ptr);      \
	decl cmp = ptrauth_sign_unauthenticated(ptr, key, \
	        ptrauth_blend_discriminator(&ptr, ptrauth_string_discriminator(discr))); \
	if (cmp != raw) { \
	        printf("kern.run_pac_test: %s (%s) (discr=%s) is not signed as expected (%p vs %p)\n", #decl, #ptr, #discr, raw, cmp); \
	        kr = EINVAL; \
	} \
}

/*
 * Allocate the containing structure, and store a pointer to the desired member,
 * which should be subject to pointer signing.
 */
#define ALLOC_VALIDATE_DATA_PTR(structure, decl, member, discr) { \
	__typed_allocators_ignore_push \
	structure *tmp =  kalloc_data(sizeof(structure), Z_WAITOK | Z_ZERO); \
	if (!tmp) return ENOMEM; \
	tmp->member = (void*)0xffffffff41414141; \
	VALIDATE_DATA_PTR(decl, tmp->member, discr) \
	kfree_data(tmp, sizeof(structure)); \
	__typed_allocators_ignore_pop \
}

#define VALIDATE_DATA_PTR(decl, ptr, discr) VALIDATE_PTR(decl, ptr, ptrauth_key_process_independent_data, discr)

/*
 * Validate that a pointer that is supposed to be signed, is, and that the signature
 * matches based on signing key, location and discriminator
 */
static int
sysctl_run_ptrauth_data_tests SYSCTL_HANDLER_ARGS
{
	#pragma unused(arg1, arg2, oidp)

	unsigned int dummy;
	int error, changed, kr;
	error = sysctl_io_number(req, 0, sizeof(dummy), &dummy, &changed);
	if (error || !changed) {
		return error;
	}

	/* proc_t */
	ALLOC_VALIDATE_DATA_PTR(struct proc, struct proc *, p_pptr, "proc.p_pptr");
	ALLOC_VALIDATE_DATA_PTR(struct proc, struct vnode *, p_textvp, "proc.p_textvp");
	ALLOC_VALIDATE_DATA_PTR(struct proc, struct pgrp *, p_pgrp.__smr_ptr, "proc.p_pgrp");

	/* The rest of the tests live in osfmk/ */
	kr = ptrauth_data_tests();

	if (error == 0) {
		error = mach_to_bsd_errno(kr);
	}

	return kr;
}

SYSCTL_PROC(_kern, OID_AUTO, run_ptrauth_data_tests,
    CTLTYPE_INT | CTLFLAG_RW | CTLFLAG_LOCKED | CTLFLAG_MASKED,
    0, 0, sysctl_run_ptrauth_data_tests, "I", "");

#endif /*  __has_feature(ptrauth_calls) */
#endif /* DEVELOPMENT || DEBUG */

