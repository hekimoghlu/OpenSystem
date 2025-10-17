/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 22, 2023.
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
#include "config.h"
#include "PASReportCrashPrivate.h"
#include <wtf/Compiler.h>

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_BEGIN

#if !USE(SYSTEM_MALLOC)
#include <bmalloc/BPlatform.h>
#if BENABLE(LIBPAS)
#include <bmalloc/pas_report_crash.h>
#endif
#endif

WTF_IGNORE_WARNINGS_IN_THIRD_PARTY_CODE_END

using namespace JSC;

#ifdef __APPLE__
kern_return_t PASReportCrashExtractResults(vm_address_t fault_address, mach_vm_address_t pas_dead_root, unsigned version, task_t task, pas_report_crash_pgm_report *report, crash_reporter_memory_reader_t crm_reader)
{
#if !USE(SYSTEM_MALLOC)
#if BENABLE(LIBPAS)
    return pas_report_crash_extract_pgm_failure(fault_address, pas_dead_root, version, task, report, crm_reader);
#endif
#endif
    UNUSED_PARAM(fault_address);
    UNUSED_PARAM(pas_dead_root);
    UNUSED_PARAM(version);
    UNUSED_PARAM(task);
    UNUSED_PARAM(report);
    UNUSED_PARAM(crm_reader);

    return KERN_FAILURE;
}
#endif /* __APPLE__ */
