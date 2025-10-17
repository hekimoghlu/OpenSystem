/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 21, 2022.
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
#ifndef __MALLOC_TYPE_INTERNAL_H__
#define __MALLOC_TYPE_INTERNAL_H__

#include <malloc/_ptrcheck.h>
__ptrcheck_abi_assume_single()

#if MALLOC_TARGET_64BIT

MALLOC_ALWAYS_INLINE MALLOC_INLINE
static malloc_type_descriptor_t
malloc_get_tsd_type_descriptor(void)
{
	return (malloc_type_descriptor_t){
		.storage = __unsafe_forge_single(void *, _pthread_getspecific_direct(__TSD_MALLOC_TYPE_DESCRIPTOR)),
	};
}

MALLOC_ALWAYS_INLINE MALLOC_INLINE
static uint64_t
malloc_get_tsd_type_id(void)
{
	return malloc_get_tsd_type_descriptor().type_id;
}

MALLOC_ALWAYS_INLINE MALLOC_INLINE
static void
malloc_set_tsd_type_descriptor(malloc_type_descriptor_t type_desc)
{
	_pthread_setspecific_direct(__TSD_MALLOC_TYPE_DESCRIPTOR,
			type_desc.storage);
}

MALLOC_ALWAYS_INLINE MALLOC_INLINE
static malloc_type_descriptor_t
malloc_callsite_fallback_type_descriptor(void)
{
	uint32_t bits =
			(uint32_t)(((uintptr_t)__builtin_return_address(0)) >> 2);
	return (malloc_type_descriptor_t){
		.hash = bits,
		// no summary bits
	};
}

MALLOC_ALWAYS_INLINE MALLOC_INLINE
static malloc_type_id_t
malloc_callsite_fallback_type_id(void)
{
	return malloc_callsite_fallback_type_descriptor().type_id;
}

// TODO: can we get a guarantee from the compiler that no valid type ID will
// ever have this value?
#define MALLOC_TYPE_ID_NONE 0ull
#define MALLOC_TYPE_ID_NONZERO 1ull
#define MALLOC_TYPE_DESCRIPTOR_NONE (malloc_type_descriptor_t){ 0 }

union malloc_type_layout_bits_u {
	uint16_t bits;
	malloc_type_layout_semantics_t layout;
};

MALLOC_ALWAYS_INLINE MALLOC_INLINE
static bool
malloc_type_descriptor_is_pure_data(malloc_type_descriptor_t type_desc)
{
	static const union malloc_type_layout_bits_u data_layout = {
		.layout = { .generic_data = true, },
	};
	union malloc_type_layout_bits_u type_desc_layout = {
		.layout = type_desc.summary.layout_semantics,
	};
	return type_desc_layout.bits == data_layout.bits;
}

MALLOC_ALWAYS_INLINE MALLOC_INLINE
static bool
malloc_type_descriptor_is_uninferred(malloc_type_descriptor_t type_desc)
{
	return type_desc.summary.type_kind == MALLOC_TYPE_KIND_C &&
			type_desc.summary.callsite_flags == MALLOC_TYPE_CALLSITE_FLAGS_NONE;
}

#else // MALLOC_TARGET_64BIT

MALLOC_ALWAYS_INLINE MALLOC_INLINE
static uint64_t
malloc_get_tsd_type_id(void)
{
	return 0;
}

typedef uint64_t malloc_type_descriptor_t;

MALLOC_ALWAYS_INLINE MALLOC_INLINE
static malloc_type_descriptor_t
malloc_callsite_fallback_type_descriptor(void)
{
	return 0;
}

MALLOC_ALWAYS_INLINE MALLOC_INLINE
static malloc_type_id_t
malloc_callsite_fallback_type_id(void)
{
	return 0;
}

#endif // MALLOC_TARGET_64BIT

#if !MALLOC_TARGET_EXCLAVES
MALLOC_NOEXPORT
extern bool malloc_interposition_compat;
#else
static const bool malloc_interposition_compat = false;
#endif // !MALLOC_TARGET_EXCLAVES

MALLOC_NOEXPORT
void
_malloc_detect_interposition(void);

#endif // __MALLOC_TYPE_INTERNAL_H__
