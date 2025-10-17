/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 30, 2025.
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
#ifndef _ARM_CACHES_INTERNAL
#define _ARM_CACHES_INTERNAL    1

#include <arm64/proc_reg.h>

#include <kern/kern_types.h>

extern void flush_dcache_syscall( vm_offset_t addr, unsigned length);

#ifdef MACH_KERNEL_PRIVATE

extern void enable_dc_mva_ops(void);
extern void disable_dc_mva_ops(void);

extern void flush_dcache(vm_offset_t addr, unsigned count, int phys);
extern void flush_dcache64(addr64_t addr, unsigned count, int phys);
extern void invalidate_icache(vm_offset_t addr, unsigned cnt, int phys);
extern void invalidate_icache64(addr64_t addr, unsigned cnt, int phys);

#if     defined(ARMA7)
#define LWFlush 1
#define LWClean 2
extern void cache_xcall(unsigned int op);
extern void cache_xcall_handler(unsigned int op);
#endif
#endif
extern void clean_dcache(vm_offset_t addr, unsigned count, int phys);
extern void clean_dcache64(addr64_t addr, unsigned count, int phys);

extern void CleanPoC_Dcache(void);
extern void CleanPoU_Dcache(void);

/*
 * May not actually perform a flush on platforms
 * where AP caches are snooped by all agents on SoC.
 *
 * This is the one you need unless you really know what
 * you're doing.
 */
extern void CleanPoC_DcacheRegion(vm_offset_t va, size_t length);

/*
 * Always actually flushes the cache, even on platforms
 * where AP caches are snooped by all agents.  You
 * probably don't need to use this.  Intended for use in
 * panic save routine (where caches will be yanked by reset
 * and coherency doesn't help).
 */
extern void CleanPoC_DcacheRegion_Force(vm_offset_t va, size_t length);

extern void CleanPoU_DcacheRegion(vm_offset_t va, size_t length);

extern void FlushPoC_Dcache(void);
extern void FlushPoU_Dcache(void);
extern void FlushPoC_DcacheRegion(vm_offset_t va, size_t length);

extern void InvalidatePoU_Icache(void);
extern void InvalidatePoU_IcacheRegion(vm_offset_t va, size_t length);

extern void cache_sync_page(ppnum_t pp);

extern void platform_cache_init(void);
extern void platform_cache_idle_enter(void);
extern void platform_cache_flush(void);
extern boolean_t platform_cache_batch_wimg(unsigned int new_wimg, unsigned int size);
extern void platform_cache_flush_wimg(unsigned int new_wimg);
extern void platform_cache_clean(void);
extern void platform_cache_shutdown(void);
extern void platform_cache_disable(void);

#endif /* #ifndef _ARM_CACHES_INTERNAL */
