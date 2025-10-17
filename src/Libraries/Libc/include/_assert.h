/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 7, 2025.
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
#ifndef __ASSERT_H_
#define __ASSERT_H_

#include <sys/cdefs.h>
#include <_bounds.h>

_LIBC_SINGLE_BY_DEFAULT()

#ifndef UNIFDEF_DRIVERKIT
#ifndef __GNUC__

#ifndef __cplusplus
#include <_abort.h>
#endif /* !__cplusplus */
#include <_printf.h>

#define __assert(e, file, line) \
	((void)printf ("%s:%d: failed assertion `%s'\n", file, line, e), abort())

#else /* __GNUC__ */

__BEGIN_DECLS
void __assert_rtn(const char *, const char *, int, const char *) __dead2 __cold __disable_tail_calls;
#if defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) && ((__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__-0) < 1070)
void __eprintf(const char *, const char *, unsigned, const char *) __dead2 __cold;
#endif
__END_DECLS

#if defined(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__) && ((__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__-0) < 1070)
#define __assert(e, file, line) \
	__eprintf ("%s:%d: failed assertion `%s'\n", file, line, e)
#else
/* 8462256: modified __assert_rtn() replaces deprecated __eprintf() */
#define __assert(e, file, line) \
	__assert_rtn (__unsafe_forge_null_terminated(const char *, -1L), file, line, e)
#endif

#endif /* __GNUC__ */
#else /* UNIFDEF_DRIVERKIT */
__BEGIN_DECLS
void __assert_rtn(const char *, const char *, int, const char *) __dead2 __cold __disable_tail_calls;
__END_DECLS
#endif /* UNIFDEF_DRIVERKIT */

#endif /* __ASSERT_H_ */
