/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 29, 2025.
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
/*
 * $FreeBSD: /repoman/r/ncvs/src/include/timeconv.h,v 1.2 2002/08/21 16:19:55 mike Exp $
 */

#ifndef _TIMECONV_H_
#define	_TIMECONV_H_

#include <sys/cdefs.h>
#include <sys/types.h>

__BEGIN_DECLS
time_t _time32_to_time(int32_t t32);
int32_t _time_to_time32(time_t t);
time_t _time64_to_time(int64_t t64);
int64_t _time_to_time64(time_t t);
long _time_to_long(time_t t);
time_t _long_to_time(long tlong);
int _time_to_int(time_t t);
time_t _int_to_time(int tint);
__END_DECLS

#endif /* _TIMECONV_H_ */
