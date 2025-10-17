/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 20, 2023.
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
#ifndef _CUPS_DEBUG_PRIVATE_H_
#  define _CUPS_DEBUG_PRIVATE_H_


/*
 * Include necessary headers...
 */

#  include <cups/versioning.h>


/*
 * C++ magic...
 */

#  ifdef __cplusplus
extern "C" {
#  endif /* __cplusplus */


/*
 * The debug macros are used if you compile with DEBUG defined.
 *
 * Usage:
 *
 *   DEBUG_set("logfile", "level", "filter", 1)
 *
 * The DEBUG_set macro allows an application to programmatically enable (or
 * disable) debug logging.  The arguments correspond to the CUPS_DEBUG_LOG,
 * CUPS_DEBUG_LEVEL, and CUPS_DEBUG_FILTER environment variables.  The 1 on the
 * end forces the values to override the environment.
 */

#  ifdef DEBUG
#    define DEBUG_set(logfile,level,filter) _cups_debug_set(logfile,level,filter,1)
#  else
#    define DEBUG_set(logfile,level,filter)
#  endif /* DEBUG */


/*
 * Prototypes...
 */

extern void	_cups_debug_set(const char *logfile, const char *level, const char *filter, int force) _CUPS_PRIVATE;
#  ifdef _WIN32
extern int	_cups_gettimeofday(struct timeval *tv, void *tz) _CUPS_PRIVATE;
#    define gettimeofday(a,b) _cups_gettimeofday(a, b)
#  endif /* _WIN32 */

#  ifdef __cplusplus
}
#  endif /* __cplusplus */

#endif /* !_CUPS_DEBUG_PRIVATE_H_ */
