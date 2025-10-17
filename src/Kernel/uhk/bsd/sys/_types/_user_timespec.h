/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 30, 2022.
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
/* LP64 version of struct timespec.  time_t is a long and must grow when
 * we're dealing with a 64-bit process.
 * WARNING - keep in sync with struct timespec
 */

#ifndef _STRUCT_USER_TIMESPEC
#define _STRUCT_USER_TIMESPEC   struct user_timespec
_STRUCT_USER_TIMESPEC
{
	user_time_t     tv_sec;         /* seconds */
	user_long_t     tv_nsec;        /* and nanoseconds */
};
#endif /* _STRUCT_USER_TIMESPEC */
#endif /* KERNEL */
