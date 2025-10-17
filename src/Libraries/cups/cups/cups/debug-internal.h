/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 30, 2021.
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
#ifndef _CUPS_DEBUG_INTERNAL_H_
#  define _CUPS_DEBUG_INTERNAL_H_


/*
 * Include necessary headers...
 */

#  include "debug-private.h"


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
 *   DEBUG_puts("string")
 *   DEBUG_printf(("format string", arg, arg, ...));
 *
 * Note the extra parenthesis around the DEBUG_printf macro...
 *
 * Newlines are not required on the end of messages, as both add one when
 * writing the output.
 *
 * If the first character is a digit, then it represents the "log level" of the
 * message from 0 to 9.  The default level is 1.  The following defines the
 * current levels we use:
 *
 * 0 = public APIs, other than value accessor functions
 * 1 = return values for public APIs
 * 2 = public value accessor APIs, progress for public APIs
 * 3 = return values for value accessor APIs
 * 4 = private APIs, progress for value accessor APIs
 * 5 = return values for private APIs
 * 6 = progress for private APIs
 * 7 = static functions
 * 8 = return values for static functions
 * 9 = progress for static functions
 */

#  ifdef DEBUG
#    define DEBUG_puts(x) _cups_debug_puts(x)
#    define DEBUG_printf(x) _cups_debug_printf x
#  else
#    define DEBUG_puts(x)
#    define DEBUG_printf(x)
#  endif /* DEBUG */


/*
 * Prototypes...
 */

#  ifdef DEBUG
extern int	_cups_debug_fd _CUPS_INTERNAL;
extern int	_cups_debug_level _CUPS_INTERNAL;
extern void	_cups_debug_printf(const char *format, ...) _CUPS_FORMAT(1,2) _CUPS_INTERNAL;
extern void	_cups_debug_puts(const char *s) _CUPS_INTERNAL;
#  endif /* DEBUG */

#  ifdef __cplusplus
}
#  endif /* __cplusplus */

#endif /* !_CUPS_DEBUG_INTERNAL_H_ */
