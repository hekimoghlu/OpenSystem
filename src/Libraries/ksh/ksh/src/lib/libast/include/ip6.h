/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 10, 2022.
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
#pragma prototyped

#ifndef _IP6_H
#define _IP6_H		1

#define IP6ADDR		16
#define IP6BITS		IP6ADDR
#define IP6PREFIX	(IP6ADDR+1)

#if _BLD_ast && defined(__EXPORT__)
#define extern		__EXPORT__
#endif

extern char*	fmtip6(const unsigned char*, int);
extern int	strtoip6(const char*, char**, unsigned char*, unsigned char*);

#undef		extern

#endif
