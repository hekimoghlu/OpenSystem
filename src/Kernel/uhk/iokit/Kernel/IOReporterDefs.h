/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 8, 2022.
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
#ifndef _IOEPORTERDEFS_H
#define _IOEPORTERDEFS_H

//#include "IOReportHubCommon.h"

//#define IORDEBUG_IOLOG

#if defined(IORDEBUG_IOLOG)
#define IORLOG(fmt, args...)    \
do {                            \
    IOLog((fmt), ##args);         \
    IOLog("\n");                \
} while(0)

#else
#define IORLOG(fmt, args...)
#endif

#define IORERROR_LOG

#ifdef IORERROR_LOG
#define IORERROR(fmt, args...) IOLog(fmt, ##args);
#else
#define IORERROR(fmt, args...)
#endif

// overflow detection routines
#if (SIZE_T_MAX < INT_MAX)
#error "(SIZE_T_MAX < INT_MAX) -> PREFL_MEMOP_*()) unsafe for size_t"
#endif

#define PREFL_MEMOP_FAIL(__val, __type) do {  \
    if (__val <= 0) {  \
	IORERROR("%s - %s <= 0!\n", __func__, #__val);  \
	res = kIOReturnUnderrun;  \
	goto finish;  \
    }  else if (__val > INT_MAX / (int)sizeof(__type)) {  \
	IORERROR("%s - %s > INT_MAX / sizeof(%s)!\n",__func__,#__val,#__type);\
	res = kIOReturnOverrun;  \
	goto finish;  \
    }  \
} while(0)

#define PREFL_MEMOP_PANIC(__val, __type) do {  \
    if (__val <= 0) {  \
	panic("%s - %s <= 0!", __func__, #__val);  \
    }  else if (__val > INT_MAX / (int)sizeof(__type)) {  \
	panic("%s - %s > INT_MAX / sizeof(%s)!", __func__, #__val, #__type);  \
    }  \
} while(0)

//#include "IOReportHubCommon.h"//



#define IOREPORTER_DEBUG_ELEMENT(idx)                                   \
do {                                                                    \
IOLog("IOReporter::DrvID: %llx | Elt:[%3d] |ID: %llx |Ticks: %llu |",   \
_elements[idx].provider_id,                                             \
idx,                                                                    \
_elements[idx].channel_id,                                              \
_elements[idx].timestamp);                                              \
IOLog("0: %llu | 1: %llu | 2: %llu | 3: %llu\n",                        \
_elements[idx].values.v[0],                                             \
_elements[idx].values.v[1],                                             \
_elements[idx].values.v[2],                                             \
_elements[idx].values.v[3]);                                            \
} while(0)


#define IOREPORTER_CHECK_LOCK()                                         \
do {                                                                    \
    if (!_reporterIsLocked) {                                           \
	panic("%s was called out of locked context!", __PRETTY_FUNCTION__); \
    }                                                                   \
} while(0)                                                              \

#define IOREPORTER_CHECK_CONFIG_LOCK()                                  \
do {                                                                    \
    if (!_reporterConfigIsLocked) {                                     \
	panic("%s was called out of config locked context!", __PRETTY_FUNCTION__); \
    }                                                                   \
} while(0)                                                              \

#endif /* ! _IOEPORTERDEFS_H */
