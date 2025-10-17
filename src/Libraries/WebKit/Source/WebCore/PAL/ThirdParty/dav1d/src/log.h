/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 27, 2025.
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
#ifndef DAV1D_SRC_LOG_H
#define DAV1D_SRC_LOG_H

#include "config.h"

#include <stdarg.h>

#include "dav1d/dav1d.h"

#include "common/attributes.h"

#if CONFIG_LOG
#define dav1d_log dav1d_log
void dav1d_log_default_callback(void *cookie, const char *format, va_list ap);
void dav1d_log(Dav1dContext *c, const char *format, ...) ATTR_FORMAT_PRINTF(2, 3);
#else
#define dav1d_log_default_callback NULL
#define dav1d_log(...) do { } while(0)
#endif

#endif /* DAV1D_SRC_LOG_H */
