/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 14, 2024.
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
/*
 * @OSF_COPYRIGHT@
 */

#ifdef  XNU_KERNEL_PRIVATE

#ifndef _KERN_CPU_DATA_H_
#define _KERN_CPU_DATA_H_

#include <mach/mach_types.h>
#include <sys/cdefs.h>

#ifdef  MACH_KERNEL_PRIVATE

#include <machine/cpu_data.h>

#endif  /* MACH_KERNEL_PRIVATE */

__BEGIN_DECLS

extern void             _disable_preemption(void);
extern void             _disable_preemption_without_measurements(void);
extern void             _enable_preemption(void);

#ifndef MACHINE_PREEMPTION_MACROS
#define disable_preemption()                    _disable_preemption()
#define disable_preemption_without_measurements() _disable_preemption_without_measurements()
#define enable_preemption()                     _enable_preemption()
#endif

#if SCHED_HYGIENE_DEBUG
#define SCHED_HYGIENE_MARKER (1u << 31)
extern void abandon_preemption_disable_measurement(void);
#endif

__END_DECLS

#endif  /* _KERN_CPU_DATA_H_ */

#endif  /* XNU_KERNEL_PRIVATE */
