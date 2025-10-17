/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 22, 2024.
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
#include <sys/cdefs.h>
#ifdef __cplusplus
#include <stdlib.h>
#endif /* __cplusplus */

/*
 * Unlike other ANSI header files, <assert.h> may usefully be included
 * multiple times, with and without NDEBUG defined.
 */

#undef assert

#ifdef NDEBUG
#define	assert(e)	((void)0)
#else

#include <_assert.h>

#ifdef __FILE_NAME__
#define __ASSERT_FILE_NAME __FILE_NAME__
#else /* __FILE_NAME__ */
#define __ASSERT_FILE_NAME __FILE__
#endif /* __FILE_NAME__ */

#ifndef UNIFDEF_DRIVERKIT
#ifndef __GNUC__

#define assert(e)  \
    ((void) ((e) ? ((void)0) : __assert (#e, __ASSERT_FILE_NAME, __LINE__)))

#else /* __GNUC__ */

#if __DARWIN_UNIX03
#define	assert(e) \
    (__builtin_expect(!(e), 0) ? __assert_rtn(__func__, __ASSERT_FILE_NAME, __LINE__, #e) : (void)0)
#else /* !__DARWIN_UNIX03 */
#define assert(e)  \
    (__builtin_expect(!(e), 0) ? __assert (#e, __ASSERT_FILE_NAME, __LINE__) : (void)0)
#endif /* __DARWIN_UNIX03 */

#endif /* __GNUC__ */
#else /* UNIFDEF_DRIVERKIT */
#define	assert(e) \
    (__builtin_expect(!(e), 0) ? __assert_rtn(__func__, __ASSERT_FILE_NAME, __LINE__, #e) : (void)0)
#endif /* UNIFDEF_DRIVERKIT */
#endif /* NDEBUG */

#include <_static_assert.h>
