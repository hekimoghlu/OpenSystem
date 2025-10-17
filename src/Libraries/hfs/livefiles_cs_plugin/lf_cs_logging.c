/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 2, 2022.
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

//
// Copyright (c) 2019-2019 Apple Inc. All rights reserved.
//
// lf_cs_logging.c - Implemenents routines for logging info, erros, warnings
//                   and debug for livefiles Apple_CoreStorage plugin.
//

#include <stdio.h>
#include <stdarg.h>

#include "lf_cs_logging.h"

#if !LF_CS_USE_OSLOG

#define VPRINTF(fmt, val)       vfprintf(stderr, fmt, val)

void
log_debug(const char *fmt, ...)
{
	va_list va;

	va_start(va, fmt);
	VPRINTF(fmt, va);
	va_end(va);
}

void
log_info(const char *fmt, ...)
{
	va_list va;

	va_start(va, fmt);
	VPRINTF(fmt, va);
	va_end(va);
}

void
log_warn(const char *fmt, ...)
{
	va_list va;

	va_start(va, fmt);
	VPRINTF(fmt, va);
	va_end(va);
}

void
log_err(const char *fmt, ...)
{
	va_list va;

	va_start(va, fmt);
	VPRINTF(fmt, va);
	va_end(va);
}

#endif /* !LF_CS_USE_OSLOG */
