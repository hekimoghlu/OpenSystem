/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 12, 2022.
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
/**
 * This file is meant to contain all of the PPL entry points and PPL-specific
 * functionality.
 *
 * Every single function in the pmap that chooses between running a "*_ppl()" or
 * "*_internal()" function variant will be placed into this file. This file also
 * contains the ppl_handler_table, as well as a few PPL-only entry/exit helper
 * functions.
 *
 * See doc/ppl.md for more information about how these PPL entry points work.
 */
#include <kern/ledger.h>

#include <vm/vm_object_xnu.h>
#include <vm/vm_page.h>
#include <vm/vm_pageout.h>

#include <arm64/sptm/pmap/pmap_internal.h>

/**
 * PMAP_SUPPORT_PROTOTYPES() will automatically create prototypes for the
 * _internal() and _ppl() variants of a PPL entry point. It also automatically
 * generates the code for the _ppl() variant which is what is used to jump into
 * the PPL.
 *
 * See doc/ppl.md for more information about how these PPL entry points work.
 */

PMAP_SUPPORT_PROTOTYPES(
	kern_return_t,
	mapping_free_prime, (void), MAPPING_FREE_PRIME_INDEX);

/**
 * See pmap_cpu_data_init_internal()'s function header for more info.
 */
void
pmap_cpu_data_init(void)
{
	pmap_cpu_data_init_internal(cpu_number());
}

/**
 * Prime the pv_entry_t free lists with a healthy amount of objects first thing
 * during boot. These objects will be used to keep track of physical-to-virtual
 * mappings.
 */
void
mapping_free_prime(void)
{
	kern_return_t kr = KERN_FAILURE;

	kr = mapping_free_prime_internal();

	if (kr != KERN_SUCCESS) {
		panic("%s: failed, no pages available? kr=%d", __func__, kr);
	}
}

/**
 * SPTM TODO: delete this function once the SPTM pmap becomes the sole pmap implementation.
 * See pmap_ledger_verify_size_internal()'s function header for more information.
 */
__attribute__((noreturn))
void
pmap_ledger_verify_size(size_t size)
{
	panic("%s: unsupported on non-PPL systems, size=%lu", __func__, size);
	__builtin_unreachable();
}

/**
 * SPTM TODO: delete this function once the SPTM pmap becomes the sole pmap implementation.
 * See pmap_ledger_alloc_internal()'s function header for more information.
 */
ledger_t
pmap_ledger_alloc(void)
{
	panic("%s: unsupported on non-PPL systems", __func__);
	__builtin_unreachable();
}

/**
 * SPTM TODO: delete this function once the SPTM pmap becomes the sole pmap implementation.
 * See pmap_ledger_free_internal()'s function header for more information.
 */
__attribute__((noreturn))
void
pmap_ledger_free(ledger_t ledger)
{
	panic("%s: unsupported on non-PPL systems, ledger=%p", __func__, ledger);
	__builtin_unreachable();
}
