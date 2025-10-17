/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 14, 2024.
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
/* Please leave this file C-compatible. */

/* Please keep this file in sync with DataTypes.h.in */

#ifndef SUPPORT_DATATYPES_H
#define SUPPORT_DATATYPES_H

#include <IndexStoreDB_LLVMSupport/toolchain_Config_indexstoredb-prefix.h>

#define HAVE_INTTYPES_H 1
#define HAVE_STDINT_H 1
#define HAVE_UINT64_T 1
#define HAVE_U_INT64_T 1

#ifdef __cplusplus
#include <cmath>
#else
#include <math.h>
#endif

#ifdef HAVE_INTTYPES_H
#include <inttypes.h>
#endif

#ifdef HAVE_STDINT_H
#include <stdint.h>
#else
#error "Compiler must provide an implementation of stdint.h"
#endif

#ifndef _MSC_VER

/* Note that <inttypes.h> includes <stdint.h>, if this is a C99 system. */
#include <sys/types.h>

#ifdef _AIX
#include <IndexStoreDB_LLVMSupport/toolchain_Support_AIXDataTypesFix.h>
#endif

/* Handle incorrect definition of uint64_t as u_int64_t */
#ifndef HAVE_UINT64_T
#ifdef HAVE_U_INT64_T
typedef u_int64_t uint64_t;
#else
# error "Don't have a definition for uint64_t on this platform"
#endif
#endif

#else /* _MSC_VER */
#include <stdlib.h>
#include <stddef.h>
#include <sys/types.h>
#ifdef __cplusplus
#include <cmath>
#else
#include <math.h>
#endif

#if defined(_WIN64)
typedef signed __int64 ssize_t;
#else
typedef signed int ssize_t;
#endif /* _WIN64 */

#ifndef HAVE_INTTYPES_H
#define PRId64 "I64d"
#define PRIi64 "I64i"
#define PRIo64 "I64o"
#define PRIu64 "I64u"
#define PRIx64 "I64x"
#define PRIX64 "I64X"

#define PRId32 "d"
#define PRIi32 "i"
#define PRIo32 "o"
#define PRIu32 "u"
#define PRIx32 "x"
#define PRIX32 "X"
#endif /* HAVE_INTTYPES_H */

#endif /* _MSC_VER */

/* Set defaults for constants which we cannot find. */
#if !defined(INT64_MAX)
# define INT64_MAX 9223372036854775807LL
#endif
#if !defined(INT64_MIN)
# define INT64_MIN ((-INT64_MAX)-1)
#endif
#if !defined(UINT64_MAX)
# define UINT64_MAX 0xffffffffffffffffULL
#endif

#ifndef HUGE_VALF
#define HUGE_VALF (float)HUGE_VAL
#endif

#endif  /* SUPPORT_DATATYPES_H */
