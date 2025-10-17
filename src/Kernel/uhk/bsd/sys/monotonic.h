/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 30, 2024.
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
#ifndef SYS_MONOTONIC_H
#define SYS_MONOTONIC_H

#include <stdbool.h>
#include <stdint.h>
#include <sys/cdefs.h>

#if !MACH_KERNEL_PRIVATE

#include <sys/ioccom.h>

__BEGIN_DECLS

/*
 * XXX These declarations are subject to change at any time.
 */

#define MT_IOC(x) _IO('m', (x))
#define MT_IOC_RESET MT_IOC(0)
#define MT_IOC_ADD MT_IOC(1)
#define MT_IOC_ENABLE MT_IOC(2)
#define MT_IOC_COUNTS MT_IOC(3)
#define MT_IOC_GET_INFO MT_IOC(4)

__END_DECLS

#endif /* !MACH_KERNEL_PRIVATE */

__BEGIN_DECLS

struct monotonic_config {
	uint64_t event;
	uint64_t allowed_ctr_mask;
	uint64_t cpu_mask;
};

union monotonic_ctl_add {
	struct {
		struct monotonic_config config;
	} in;

	struct {
		uint32_t ctr;
	} out;
};

union monotonic_ctl_enable {
	struct {
		bool enable;
	} in;
};


union monotonic_ctl_counts {
	struct {
		uint64_t ctr_mask;
	} in;

	struct {
		uint64_t counts[1];
	} out;
};


union monotonic_ctl_info {
	struct {
		unsigned int nmonitors;
		unsigned int ncounters;
	} out;
};

__END_DECLS

#if XNU_KERNEL_PRIVATE

#if CONFIG_CPU_COUNTERS

#include <kern/monotonic.h>
#include <machine/monotonic.h>
#include <sys/kdebug.h>
#include <kern/locks.h>

__BEGIN_DECLS

/*
 * MT_KDBG_TMP* macros are meant for temporary (i.e. not checked-in)
 * performance investigations.
 */

/*
 * Record the current CPU counters.
 *
 * Preemption must be disabled.
 */
#define MT_KDBG_TMPCPU_EVT(CODE) \
	KDBG_EVENTID(DBG_MONOTONIC, DBG_MT_TMPCPU, CODE)

#define MT_KDBG_TMPCPU_(CODE, FUNC) \
	do { \
	        if (kdebug_enable && \
	                        kdebug_debugid_enabled(MT_KDBG_TMPCPU_EVT(CODE))) { \
	                uint64_t __counts[MT_CORE_NFIXED]; \
	                mt_fixed_counts(__counts); \
	                KDBG(MT_KDBG_TMPCPU_EVT(CODE) | (FUNC), __counts[MT_CORE_INSTRS], \
	                                __counts[MT_CORE_CYCLES]); \
	        } \
	} while (0)

#define MT_KDBG_TMPCPU(CODE) MT_KDBG_TMPCPU_(CODE, DBG_FUNC_NONE)
#define MT_KDBG_TMPCPU_START(CODE) MT_KDBG_TMPCPU_(CODE, DBG_FUNC_START)
#define MT_KDBG_TMPCPU_END(CODE) MT_KDBG_TMPCPU_(CODE, DBG_FUNC_END)

extern lck_grp_t mt_lock_grp;

int mt_dev_init(void);

struct mt_device {
	const char *mtd_name;
	int(*const mtd_init)(struct mt_device *dev);
	int(*const mtd_add)(struct monotonic_config *config, uint32_t *ctr_out);
	void(*const mtd_reset)(void);
	void(*const mtd_enable)(bool enable);
	int(*const mtd_read)(uint64_t ctr_mask, uint64_t *counts_out);
	decl_lck_mtx_data(, mtd_lock);

	uint8_t mtd_nmonitors;
	uint8_t mtd_ncounters;
	bool mtd_inuse;
};
typedef struct mt_device *mt_device_t;

extern struct mt_device mt_devices[];

__END_DECLS

#endif /* CONFIG_CPU_COUNTERS */

#endif /* XNU_KERNEL_PRIVATE */

#endif /* !defined(SYS_MONOTONIC_H) */
