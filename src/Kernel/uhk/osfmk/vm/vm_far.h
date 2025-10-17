/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 30, 2022.
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
#pragma once
#ifdef KERNEL_PRIVATE
#include <stdint.h>
#include <stddef.h>
#include <sys/cdefs.h>
#include <stdbool.h>
#include <kern/panic_call.h>

#ifdef __arm64__
#include <arm64/speculation.h>
#endif /* __arm64__ */

/*
 * The VM_FAR poison is set in a pointer's top 16 bits when its offset exceeds
 * the VM_FAR bounds.
 */
#define VM_FAR_POISON_VALUE (0x2BADULL)
#define VM_FAR_POISON_SHIFT (48)
#define VM_FAR_POISON_MASK (0xFFFFULL << VM_FAR_POISON_SHIFT)
#define VM_FAR_POISON_BITS (VM_FAR_POISON_VALUE << VM_FAR_POISON_SHIFT)

#define VM_FAR_ACCESSOR

__pure2
__attribute__((always_inline))
static inline void *
vm_far_add_ptr_internal(void *ptr, uint64_t idx, size_t elem_size,
    bool __unused idx_small)
{

	uintptr_t ptr_i = (uintptr_t)(ptr);
	uintptr_t new_ptr_i = ptr_i + (idx * elem_size);


	return __unsafe_forge_single(void *, new_ptr_i);
}

__attribute__((always_inline))
static inline void *
vm_far_add_ptr_bounded_fatal_unsigned_internal(void *ptr, uint64_t idx,
    size_t count, size_t elem_size, bool __unused idx_small)
{
	void *__single new_ptr = vm_far_add_ptr_internal(
		ptr, idx, elem_size,
		/*
		 * Since we're bounds checking the index, we can support small index
		 * optimizations even when the index is large.
		 */
		/* idx_small */ false);

	bool guarded_ptr_valid;
	void *__single guarded_ptr;
#if __arm64__
	/* Guard passes if idx < count */
	SPECULATION_GUARD_ZEROING_XXX(
		/* out */ guarded_ptr, /* out_valid */ guarded_ptr_valid,
		/* value */ new_ptr,
		/* cmp1 */ idx, /* cmp2 */ count,
		/* cc */ "LO");
#else
	/*
	 * We don't support guards on this target, so just perform a normal bounds
	 * check.
	 */
	guarded_ptr_valid = idx < count;
	guarded_ptr = new_ptr;
#endif /* __arm64__ */

	if (__improbable(!guarded_ptr_valid)) {
		panic("vm_far bounds check failed idx=%llu/count=%zu", idx, count);
	}

	return guarded_ptr;
}

__pure2
__attribute__((always_inline))
static inline void *
vm_far_add_ptr_bounded_poison_unsigned_internal(void *ptr, uint64_t idx,
    size_t count, size_t elem_size, bool __unused idx_small)
{
	void *__single new_ptr = vm_far_add_ptr_internal(
		ptr, idx, elem_size,
		/*
		 * Since we're bounds checking the index, we can support small index
		 * optimizations even when the index is large.
		 */
		/* idx_small */ false);

	void *__single guarded_ptr;

	/*
	 * Poison the top 16-bits with a well-known code so that later dereferences
	 * of the poisoned pointer are easy to identify.
	 */
	uintptr_t poisoned_ptr_i = (uintptr_t)new_ptr;
	poisoned_ptr_i &= ~VM_FAR_POISON_MASK;
	poisoned_ptr_i |= VM_FAR_POISON_BITS;

#if __arm64__
	SPECULATION_GUARD_SELECT_XXX(
		/* out  */ guarded_ptr,
		/* cmp1 */ idx, /* cmp2 */ count,
		/* cc   */ "LO", /* value_cc */ (uintptr_t)new_ptr,
		/* n_cc */ "HS", /* value_n_cc */ poisoned_ptr_i);
#else
	/*
	 * We don't support guards on this target, so just perform a normal bounds
	 * check.
	 */
	if (__probable(idx < count)) {
		guarded_ptr = new_ptr;
	} else {
		guarded_ptr = __unsafe_forge_single(void *, poisoned_ptr_i);
	}
#endif /* __arm64__ */

	return guarded_ptr;
}

/**
 * Compute &PTR[IDX] without enforcing VM_FAR.
 *
 * In this variant, IDX will not be bounds checked.
 */
#define VM_FAR_ADD_PTR_UNBOUNDED(ptr, idx) \
	((__typeof__((ptr))) vm_far_add_ptr_internal( \
	        (ptr), (idx), sizeof(__typeof__(*(ptr))), sizeof((idx)) <= 4))

/**
 * Compute &PTR[IDX] without enforcing VM_FAR.
 *
 * If the unsigned IDX value exceeds COUNT, trigger a panic.
 */
#define VM_FAR_ADD_PTR_BOUNDED_FATAL_UNSIGNED(ptr, idx, count) \
	((__typeof__((ptr))) vm_far_add_ptr_bounded_fatal_unsigned_internal( \
	        (ptr), (idx), (count), sizeof(__typeof__(*(ptr))), \
	        sizeof((idx)) <= 4))

/**
 * Compute &PTR[IDX] without enforcing VM_FAR.
 *
 * If the unsigned IDX value exceeds COUNT, poison the pointer such that
 * attempting to dereference it will fault.
 */
#define VM_FAR_ADD_PTR_BOUNDED_POISON_UNSIGNED(ptr, idx, count) \
	((__typeof__((ptr))) vm_far_add_ptr_bounded_poison_unsigned_internal( \
	        (ptr), (idx), (count), sizeof(__typeof__(*(ptr))), \
	        sizeof((idx)) <= 4))

#endif /* KERNEL_PRIVATE */
