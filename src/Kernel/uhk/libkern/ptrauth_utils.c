/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 20, 2022.
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
#include <IOKit/IOLib.h>
#include <kern/debug.h> // for panic()

#include <libkern/ptrauth_utils.h>

/*
 * On ptrauth systems, ptrauth_utils_sign_blob_generic is implemented
 * in osfmk/arm64/machine_routines_asm.s
 */

#if !__has_feature(ptrauth_calls)
ptrauth_generic_signature_t
ptrauth_utils_sign_blob_generic(__unused const void * ptr, __unused size_t len_bytes, __unused uint64_t data, __unused int flags)
{
	return 0;
}
#endif // __has_feature(ptrauth_calls)


/*
 * ptrauth_utils_auth_blob_generic
 *
 * Authenticate signature produced by ptrauth_utils_sign_blob_generic
 */

#if __has_feature(ptrauth_calls)
__attribute__((noinline))
void
ptrauth_utils_auth_blob_generic(const void * ptr, size_t len_bytes, uint64_t data, int flags, ptrauth_generic_signature_t signature)
{
	ptrauth_generic_signature_t calculated_signature = 0;

	if (ptr == NULL) {
		if (flags & PTRAUTH_NON_NULL) {
			panic("ptrauth_utils_auth_blob_generic: ptr must not be NULL");
		} else {
			return;
		}
	}

	if ((calculated_signature = ptrauth_utils_sign_blob_generic(ptr, len_bytes, data, flags)) == signature) {
		return;
	} else {
		panic("signature mismatch for %lu bytes at %p, calculated %lx vs %lx", len_bytes,
		    ptr,
		    calculated_signature,
		    signature);
	}
}
#else
void
ptrauth_utils_auth_blob_generic(__unused const void * ptr, __unused size_t len_bytes, __unused uint64_t data, __unused int flags, __unused ptrauth_generic_signature_t signature)
{
	return;
}
#endif // __has_feature(ptrauth_calls)
