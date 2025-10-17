/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 19, 2022.
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
#ifndef _PMAP_PCID_
#define _PMAP_PCID_     1
#if defined(__x86_64__)
void pmap_pcid_initialize(pmap_t);
void pmap_pcid_initialize_kernel(pmap_t);
pcid_t  pmap_pcid_allocate_pcid(int);
void    pmap_pcid_deallocate_pcid(int, pmap_t);
void    pmap_destroy_pcid_sync_action(void *);
void    pmap_destroy_pcid_sync(pmap_t);
void    pmap_pcid_lazy_flush(pmap_t);
void    pmap_pcid_activate(pmap_t, int, boolean_t, boolean_t);
pcid_t  pcid_for_pmap_cpu_tuple(pmap_t, thread_t, int);

#define PMAP_INVALID ((pmap_t)0xDEAD7347)
#define PMAP_PCID_INVALID_PCID  (0xDEAD)
#define PMAP_PCID_MAX_REFCOUNT (0xF0)
#define PMAP_PCID_MIN_PCID (1)

extern uint32_t pmap_pcid_ncpus;

static inline void
pmap_pcid_invalidate_all_cpus(pmap_t tpmap)
{
	unsigned i;

	pmap_assert((sizeof(tpmap->pmap_pcid_coherency_vector) >= real_ncpus) && (!(sizeof(tpmap->pmap_pcid_coherency_vector) & 7)));

	for (i = 0; i < real_ncpus; i += 8) {
		*(uint64_t *)(uintptr_t)&tpmap->pmap_pcid_coherency_vector[i] = (~0ULL);
	}
}

static inline void
pmap_pcid_validate_current(void)
{
	int     ccpu = cpu_number();
	volatile uint8_t *cptr = cpu_datap(ccpu)->cpu_pmap_pcid_coherentp;
#ifdef  PMAP_MODULE
	pmap_assert(cptr == &(current_thread()->map->pmap->pmap_pcid_coherency_vector[ccpu]));
#endif
	if (cptr) {
		*cptr = 0;
	}
}

static inline void
pmap_pcid_invalidate_cpu(pmap_t tpmap, int ccpu)
{
	tpmap->pmap_pcid_coherency_vector[ccpu] = 0xFF;
}

static inline void
pmap_pcid_validate_cpu(pmap_t tpmap, int ccpu)
{
	tpmap->pmap_pcid_coherency_vector[ccpu] = 0;
}
#endif /* x86_64 */
#endif
