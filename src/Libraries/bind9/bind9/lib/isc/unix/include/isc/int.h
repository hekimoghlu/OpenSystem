/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 5, 2025.
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
/* $Id: int.h,v 1.16 2007/06/19 23:47:19 tbox Exp $ */

#ifndef ISC_INT_H
#define ISC_INT_H 1

/*! \file */

typedef char				isc_int8_t;
typedef unsigned char			isc_uint8_t;
typedef short				isc_int16_t;
typedef unsigned short			isc_uint16_t;
typedef int				isc_int32_t;
typedef unsigned int			isc_uint32_t;
typedef long long			isc_int64_t;
typedef unsigned long long		isc_uint64_t;

#define ISC_INT8_MIN	-128
#define ISC_INT8_MAX	127
#define ISC_UINT8_MAX	255

#define ISC_INT16_MIN	-32768
#define ISC_INT16_MAX	32767
#define ISC_UINT16_MAX	65535

/*%
 * Note that "int" is 32 bits on all currently supported Unix-like operating
 * systems, but "long" can be either 32 bits or 64 bits, thus the 32 bit
 * constants are not qualified with "L".
 */
#define ISC_INT32_MIN	-2147483648
#define ISC_INT32_MAX	2147483647
#define ISC_UINT32_MAX	4294967295U

#define ISC_INT64_MIN	-9223372036854775808LL
#define ISC_INT64_MAX	9223372036854775807LL
#define ISC_UINT64_MAX	18446744073709551615ULL

#endif /* ISC_INT_H */
