/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 25, 2025.
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
#ifndef _KCOV_DATA_H_
#define _KCOV_DATA_H_

#include <san/kcov.h>
#include <san/kcov_ksancov_data.h>
#include <san/kcov_stksz_data.h>

#if KERNEL_PRIVATE

#if CONFIG_KCOV

/*
 * Coverage sanitizer per-cpu data
 */
struct kcov_cpu_data {
	uint32_t       kcd_enabled;     /* coverage recording enabled for CPU. */
};

/*
 * Coverage sanitizer per-thread data
 */
struct kcov_thread_data {
	uint32_t               ktd_disabled;    /* disable sanitizer for a thread */
#if CONFIG_KSANCOV
	ksancov_dev_t          ktd_device;     /* ksancov per-thread data */
#endif
#if CONFIG_STKSZ
	kcov_stksz_thread_t    ktd_stksz;       /* stack size per-thread data */
#endif
};

#endif /* CONFIG_KCOV */

#endif /* KERNEL_PRIVATE */

#endif /* _KCOV_DATA_H_ */
