/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 1, 2024.
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
#ifndef _SKYWALK_MEM_SKMEMVAR_H
#define _SKYWALK_MEM_SKMEMVAR_H

#ifdef BSD_KERNEL_PRIVATE
#include <skywalk/mem/skmem_region_var.h>
#include <skywalk/mem/skmem_cache_var.h>
#include <skywalk/mem/skmem_arena_var.h>

#if defined(__arm64__)
#define SKMEM_PAGE_SIZE PAGE_MAX_SIZE
#else /* __i386__, __x86_64__ */
#define SKMEM_PAGE_SIZE PAGE_SIZE
#endif /* __i386__, __x86_64__ */

/*
 * Initial segment sizes for non-monolithic regions.  Smaller size allows more
 * finer-grained control over the memory, at the expense of overheads.  The
 * effective region segment size is computed by skmem_region_params_config(),
 * depending on the number and size of objects.  This may be overridden by
 * the "skmem_seg_size" and "skmem_buf_seg_size" boot-arg.
 *
 * The sizes are always multiples of SKMEM_PAGE_SIZE.
 */
#define SKMEM_MIN_SEG_SIZE      (16 * 1024)
#define SKMEM_SEG_SIZE          SKMEM_MIN_SEG_SIZE /* default for all */
#define SKMEM_MD_SEG_SIZE       (16 * 1024)     /* default for metadata */
#define SKMEM_DRV_BUF_SEG_SIZE  (64 * 1024)     /* default for device buffer */
#define SKMEM_USR_BUF_SEG_SIZE  (16 * 1024)     /* default for user buffer */

#define SKMEM_DRV_BUF_SEG_MULTIPLIER    2

#define SKMEM_ARENA_PREFIX      "ska"
#define SKMEM_REGION_PREFIX     "skr"
#define SKMEM_CACHE_PREFIX      "skc"
#define SKMEM_ZONE_PREFIX       "skywalk"
#if DEBUG || DEVELOPMENT
#define SMKEM_KT_DEFAULT        KT_PRIV_ACCT
#else
#define SMKEM_KT_DEFAULT        KT_DEFAULT
#endif
#define SKMEM_TYPE_DEFINE(var, type) \
	KALLOC_TYPE_DEFINE(var, type, SMKEM_KT_DEFAULT)

extern lck_attr_t skmem_lock_attr;
extern lck_grp_t skmem_lock_grp;
extern uint32_t skmem_usr_buf_seg_size;

#if (DEVELOPMENT || DEBUG)
SYSCTL_DECL(_kern_skywalk_mem);
#endif /* (DEVELOPMENT || DEBUG) */

#define SKMEM_MEM_CONSTRAINED_DEVICE()    \
    (max_mem_actual <= (2ULL * 1024ULL * 1024ULL * 1024ULL))

__BEGIN_DECLS
extern void skmem_init(void);
extern void skmem_fini(void);
extern const struct skmem_region_params *skmem_get_default(skmem_region_id_t);
extern uint32_t skmem_cpu_cache_line_size(void);
extern void skmem_dispatch(thread_call_t, void (*func)(void), uint64_t);
extern struct skmem_region *skmem_get_sysctls_region(void);
extern char *skmem_dump(struct skmem_region *);
extern boolean_t skmem_lowmem_check(void);
#if (DEVELOPMENT || DEBUG)
extern void skmem_test_init(void);
extern void skmem_test_fini(void);
extern bool skmem_test_enabled(void);
#endif /* (DEVELOPMENT || DEBUG) */
__END_DECLS
#endif /* BSD_KERNEL_PRIVATE */
#endif /* _SKYWALK_MEM_SKMEMVAR_H */
