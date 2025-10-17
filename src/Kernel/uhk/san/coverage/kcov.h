/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 11, 2024.
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
#ifndef _KCOV_H_
#define _KCOV_H_

#if KERNEL_PRIVATE

#if !CONFIG_KCOV && __has_feature(coverage_sanitizer)
# error "Coverage sanitizer enabled in compiler, but kernel is not configured for KCOV"
#endif

#if CONFIG_KCOV

/* Forward declaration for types used in interfaces below. */
typedef struct kcov_cpu_data kcov_cpu_data_t;
typedef struct kcov_thread_data kcov_thread_data_t;


__BEGIN_DECLS

/* osfmk exported */
kcov_cpu_data_t *current_kcov_data(void);
kcov_cpu_data_t *cpu_kcov_data(int);

/* Init code */
void kcov_init_thread(kcov_thread_data_t *);
void kcov_start_cpu(int cpuid);

/* helpers */
void kcov_panic_disable(void);

/* per-thread */
struct kcov_thread_data *kcov_get_thread_data(thread_t);

void kcov_enable(void);
void kcov_disable(void);

/*
 * SanitizerCoverage ABI
 */
void __sanitizer_cov_pcs_init(uintptr_t *start, uintptr_t *stop);
void __sanitizer_cov_trace_pc_guard(uint32_t *guard);
void __sanitizer_cov_trace_pc_guard_init(uint32_t *start, uint32_t *stop);
void __sanitizer_cov_trace_pc_indirect(void *callee);
void __sanitizer_cov_trace_pc(void);

__END_DECLS

#endif /* CONFIG_KCOV */

#endif /* KERNEL_PRIVATE */

#endif /* _KCOV_H_ */
