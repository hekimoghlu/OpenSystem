/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 27, 2025.
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
#ifndef __LIBC_HOOKS_H__
#define __LIBC_HOOKS_H__

#include <os/base.h>
#include <sys/cdefs.h>
#include <wchar.h>

// Investigate using ptrauth_key_process_dependent_code which may provide
// additional security features for mixed arm64/arm64e processes
// rdar://123026531
#if __has_feature(ptrauth_calls)
#include <ptrauth.h>
#define __PTRAUTH_LIBC_HOOK(hook) \
	__ptrauth(ptrauth_key_process_independent_code, 1, \
			  ptrauth_string_discriminator("libc_hook_ptr." #hook)) hook
#else
#define __PTRAUTH_LIBC_HOOK(hook) hook
#endif

// This header file defines the entire interface between libc and an external
// user of libc. It's purpose is to allow an external user the ability to get
// notification whenever libc accesses the data provided to it by a client or
// caller of an API. The initial purpose of this sort of notification is for
// an address sanitizer, containing additional meta-data about memory related
// bounds and accessibility, to perform such checks and respond appropriately
// such as failing with a diagnostic message. There is only a single hook for
// each of the (presently) four entrypoints and these hooks are initially all
// empty. All of the entrypoints must be set in a single call and the opera-
// tion is destructive to any hook already registered but chaining is allowed
// but will have to managed by the setter which returns the current earlier
// set of hooks which can then be used by the setter to pre or post chain to
// as desired.

__BEGIN_DECLS

typedef unsigned long libc_hooks_version_t;
#define libc_hooks_version ((libc_hooks_version_t) 1)

typedef void (*__PTRAUTH_LIBC_HOOK(libc_hooks_will_read_t)) (const void *p, size_t size);
typedef void (*__PTRAUTH_LIBC_HOOK(libc_hooks_will_read_cstring_t)) (const char *s);
typedef void (*__PTRAUTH_LIBC_HOOK(libc_hooks_will_read_wcstring_t)) (const wchar_t *wcs);
typedef void (*__PTRAUTH_LIBC_HOOK(libc_hooks_will_write_t)) (const void *p, size_t size);

typedef struct {
	libc_hooks_version_t version;
	libc_hooks_will_read_t will_read;
	libc_hooks_will_read_cstring_t will_read_cstring;
	libc_hooks_will_read_wcstring_t will_read_wcstring;
	libc_hooks_will_write_t will_write;
} libc_hooks_t;

// Note:
//  libc_set_introspection_hooks is not thread-safe. It should
//  only be called while the process is still single-threaded.
OS_EXPORT
void
libc_set_introspection_hooks(const libc_hooks_t *new_hooks, libc_hooks_t *old_hooks, size_t size);

__END_DECLS

#endif // __LIBC_HOOKS_H__
