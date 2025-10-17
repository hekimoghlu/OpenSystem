/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 7, 2022.
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
#ifndef _KCOV_KSANCOV_DATA_H_
#define _KCOV_KSANCOV_DATA_H_

#if KERNEL_PRIVATE

#if CONFIG_KSANCOV

/*
 * Supported coverage modes.
 */
typedef enum {
	KS_MODE_NONE,
	KS_MODE_TRACE,
	KS_MODE_COUNTERS,
	KS_MODE_STKSIZE,
	KS_MODE_MAX
} ksancov_mode_t;

/*
 * A header that is always present in every ksancov mode shared memory structure.
 */
typedef struct ksancov_header {
	uint32_t         kh_magic;
	_Atomic uint32_t kh_enabled;
} ksancov_header_t;

/*
 * TRACE mode data structure.
 */

/*
 * All trace based tools share this structure.
 */
typedef struct ksancov_trace {
	ksancov_header_t kt_hdr;         /* header (must be always first) */
	uint32_t         kt_maxent;      /* Maximum entries in this shared buffer. */
	_Atomic uint32_t kt_head;        /* Pointer to the first unused element. */
	uint64_t         kt_entries[];   /* Trace entries in this buffer. */
} ksancov_trace_t;

/* PC tracing only records PCs */
typedef uintptr_t ksancov_trace_pc_ent_t;

/* STKSIZE tracing records PCs and stack size. */
typedef struct ksancov_trace_stksize_entry {
	uintptr_t pc;                      /* PC */
	uint32_t  stksize;                 /* associated stack size */
} ksancov_trace_stksize_ent_t;

/*
 * COUNTERS mode data structure.
 */
typedef struct ksancov_counters {
	ksancov_header_t kc_hdr;
	uint32_t         kc_nedges;       /* total number of edges */
	uint8_t          kc_hits[];       /* hits on each edge (8bit saturating) */
} ksancov_counters_t;

/*
 * Edge to PC mapping.
 */
typedef struct ksancov_edgemap {
	uint32_t  ke_magic;
	uint32_t  ke_nedges;
	uintptr_t ke_addrs[];             /* address of each edge relative to 'offset' */
} ksancov_edgemap_t;

/*
 * Represents state of a ksancov device when userspace asks for coverage data recording.
 */

struct ksancov_dev {
	ksancov_mode_t mode;

	union {
		ksancov_header_t       *hdr;
		ksancov_trace_t        *trace;
		ksancov_counters_t     *counters;
	};
	size_t sz;     /* size of allocated trace/counters buffer */

	size_t maxpcs;

	thread_t thread;
	dev_t dev;
	lck_mtx_t lock;
};
typedef struct ksancov_dev * ksancov_dev_t;


#endif /* CONFIG_KSANCOV */

#endif /* KERNEL_PRIVATE */

#endif /* _KCOV_KSANCOV_DATA_H_ */
