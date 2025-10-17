/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 4, 2024.
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
#ifdef _ISOC99_SOURCE
#include <stdint.h>
#endif

#if defined(_MSC_VER)
typedef __int64                  MPL_Int64;
#else
#if defined(_ISOC99_SOURCE)
typedef int64_t                  MPL_Int64;
#else
typedef long long                MPL_Int64;
#endif
#endif

#if !defined(MPL_U64)
#define MPL_U64(u) (* (MPL_Int64 *) &(u) )
#endif /* MPL_U64 */

#if !defined(MPL_isnan64)
#if !defined(_MSC_VER)
#define MPL_isnan64(u) \
  ( (( MPL_U64(u) & 0x7ff0000000000000LL)  == 0x7ff0000000000000LL)  && ((MPL_U64(u) &  0x000fffffffffffffLL) != 0)) ? 1:0
#else
#define MPL_isnan64(u) \
  ( (( MPL_U64(u) & 0x7ff0000000000000i64) == 0x7ff0000000000000i64)  && ((MPL_U64(u) & 0x000fffffffffffffi64) != 0)) ? 1:0
#endif
#endif /* MPL_isnan64 */

#if !defined(MPL_isinf64)
#if !defined(_MSC_VER)
#define MPL_isinf64(u) \
  ( (( MPL_U64(u) & 0x7ff0000000000000LL)  == 0x7ff0000000000000LL)  && ((MPL_U64(u) &  0x000fffffffffffffLL) == 0)) ? 1:0
#else
#define MPL_isinf64(u) \
  ( (( MPL_U64(u) & 0x7ff0000000000000i64) == 0x7ff0000000000000i64)  && ((MPL_U64(u) & 0x000fffffffffffffi64) == 0)) ? 1:0
#endif
#endif /* MPL_isinf64 */

#if !defined(MPL_isfinite64)
#if !defined(_MSC_VER)
#define MPL_isfinite64(u) \
  ( (( MPL_U64(u) & 0x7ff0000000000000LL)  != 0x7ff0000000000000LL)) ? 1:0
#else
#define MPL_isfinite64(u) \
  ( (( MPL_U64(u) & 0x7ff0000000000000i64) != 0x7ff0000000000000i64)) ? 1:0
#endif
#endif /* MPL_isfinite64 */

#if !defined(MPL_notisfinite64)
#if !defined(_MSC_VER)
#define MPL_notisfinite64(u) \
  ( (( MPL_U64(u) & 0x7ff0000000000000LL)  == 0x7ff0000000000000LL)) ? 1:0
#else
#define MPL_notisfinite64(u) \
  ( (( MPL_U64(u) & 0x7ff0000000000000i64) == 0x7ff0000000000000i64)) ? 1:0
#endif
#endif /* MPL_notisfinite64 */


