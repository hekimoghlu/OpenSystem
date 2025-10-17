/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 27, 2024.
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
#pragma once

#include <JavaScriptCore/JSBase.h>

#ifdef __APPLE__
#include <bmalloc/pas_report_crash_pgm_report.h>

#ifdef __cplusplus
extern "C" {
#endif

JS_EXPORT kern_return_t PASReportCrashExtractResults(vm_address_t fault_address, mach_vm_address_t pas_dead_root, unsigned version, task_t task, pas_report_crash_pgm_report *report, crash_reporter_memory_reader_t crm_reader);

#ifdef __cplusplus
}
#endif

#endif /* __APPLE__ */
