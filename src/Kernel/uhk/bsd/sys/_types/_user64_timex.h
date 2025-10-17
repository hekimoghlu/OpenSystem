/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 5, 2023.
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
#ifdef KERNEL
#ifndef _STRUCT_USER64_TIMEX
#define _STRUCT_USER64_TIMEX    struct user64_timex
_STRUCT_USER64_TIMEX
{
	u_int32_t modes;
	user64_long_t   offset;
	user64_long_t   freq;
	user64_long_t   maxerror;
	user64_long_t   esterror;
	__int32_t       status;
	user64_long_t   constant;
	user64_long_t   precision;
	user64_long_t   tolerance;

	user64_long_t   ppsfreq;
	user64_long_t   jitter;
	__int32_t       shift;
	user64_long_t   stabil;
	user64_long_t   jitcnt;
	user64_long_t   calcnt;
	user64_long_t   errcnt;
	user64_long_t   stbcnt;
};
#endif /* _STRUCT_USER64_TIMEX */
#endif /* KERNEL */
