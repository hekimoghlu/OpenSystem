/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 15, 2022.
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
#ifndef PAS_REPORT_CRASH_PGM_REPORT_H
#define PAS_REPORT_CRASH_PGM_REPORT_H

/* This file exposes a SPI between OSAnalytics and libpas ultimately called through
 * JavaScriptCore. Upon crashing of a process, on Apple platforms, ReportCrash will call
 * into libpas (through JSC) to inspect whether it was a PGM crash in libpas or not. We will report
 * back results from libpas with any information about the PGM crash. This will be logged in
 * the local crash report logs generated on the device. */

#ifdef __APPLE__
#include <mach/mach_types.h>
#include <mach/vm_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Read memory from crashed process. */
typedef void *(*crash_reporter_memory_reader_t)(task_t task, vm_address_t address, size_t size);

/* Crash Report Version number. This must be in sync between ReportCrash and libpas to generate a report. */
const unsigned pas_crash_report_version = 2;

/* Report sent back to the ReportCrash process. */
typedef struct {
    const char *error_type;
    const char *confidence;
    const char *alignment;
    vm_address_t fault_address;
    size_t allocation_size;
} pas_report_crash_pgm_report;
#endif /* __APPLE__ */

#ifdef __cplusplus
}
#endif

#endif /* PAS_REPORT_CRASH_PGM_REPORT_H */
