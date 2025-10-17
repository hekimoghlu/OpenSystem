/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 10, 2024.
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
#ifndef PAS_REPORT_CRASH_H
#define PAS_REPORT_CRASH_H

/* These functions should never be called directly; they should instead use the SPI
 * defined in JavaScriptCore PasReportCrashPrivate.h. */

#ifdef __APPLE__
#include "pas_report_crash_pgm_report.h"
#include "pas_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

PAS_API kern_return_t pas_report_crash_extract_pgm_failure(vm_address_t fault_address, mach_vm_address_t pas_dead_root, unsigned version, task_t, pas_report_crash_pgm_report*, crash_reporter_memory_reader_t crm_reader);

#ifdef __cplusplus
}
#endif

#endif /* __APPLE__ */
#endif /* pas_report_crash_h */
