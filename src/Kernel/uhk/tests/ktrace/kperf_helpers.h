/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 10, 2024.
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

#ifndef KPERF_HELPERS_H
#define KPERF_HELPERS_H

#include <unistd.h>
#include <stdbool.h>

void configure_kperf_stacks_timer(pid_t pid, unsigned int period_ms,
    bool quiet);

#define PERF_SAMPLE KDBG_EVENTID(DBG_PERF, 0, 0)
#define PERF_KPC_PMI KDBG_EVENTID(DBG_PERF, 6, 0)
#define PERF_STK_KHDR  UINT32_C(0x25020014)
#define PERF_STK_UHDR  UINT32_C(0x25020018)
#define PERF_STK_KDATA UINT32_C(0x2502000c)
#define PERF_STK_UDATA UINT32_C(0x25020010)

#define PERF_STK_EXHdr      UINT32_C(0x25020028)
#define PERF_STK_EXSample   UINT32_C(0x2502002c)
#define PERF_STK_EXStackHdr UINT32_C(0x25020030)
#define PERF_STK_EXStack    UINT32_C(0x25020034)
#define PERF_STK_KEXOffset  UINT32_C(0x25020038)

#endif // !defined(KPERF_HELPERS_H)
