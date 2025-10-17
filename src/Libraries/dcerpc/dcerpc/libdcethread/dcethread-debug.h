/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 14, 2023.
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
#ifndef __PTHREAD_WRAP_DEBUG_H__
#define __PTHREAD_WRAP_DEBUG_H__

#define DCETHREAD_DEBUG_ERROR (0)
#define DCETHREAD_DEBUG_WARNING (1)
#define DCETHREAD_DEBUG_INFO (2)
#define DCETHREAD_DEBUG_VERBOSE (3)
#define DCETHREAD_DEBUG_TRACE (4)

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>

#define DCETHREAD_DEBUG(level, ...) (dcethread__debug_printf(__FILE__, __LINE__, level, __VA_ARGS__))
#define DCETHREAD_ERROR(...) DCETHREAD_DEBUG(DCETHREAD_DEBUG_ERROR, __VA_ARGS__)
#define DCETHREAD_WARNING(...) DCETHREAD_DEBUG(DCETHREAD_DEBUG_WARNING, __VA_ARGS__)
#define DCETHREAD_INFO(...) DCETHREAD_DEBUG(DCETHREAD_DEBUG_INFO, __VA_ARGS__)
#define DCETHREAD_VERBOSE(...) DCETHREAD_DEBUG(DCETHREAD_DEBUG_VERBOSE, __VA_ARGS__)
#define DCETHREAD_TRACE(...) DCETHREAD_DEBUG(DCETHREAD_DEBUG_TRACE, __VA_ARGS__)

void dcethread__debug_set_callback(void (*cb) (const char*, unsigned int, int, const char*, void*), void* data);
void dcethread__debug_printf(const char* file, unsigned int line, int level, const char* fmt, ...)
#if __GNUC__
__attribute__((__format__ (__printf__, 4, 5)))
#endif
;

void dcethread__default_log_callback (const char* file, unsigned int line, int level, const char* str, void* data);

#endif
