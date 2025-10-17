/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 24, 2023.
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
 * CGA.h
 * - Cryptographically Generated Addresses support routines
 */

/* 
 * Modification History
 *
 * April 11, 2013 	Dieter Siegmund (dieter@apple.com)
 * - initial revision
 */

#ifndef _S_CGA_H
#define _S_CGA_H

#include <net/if.h>
#include <netinet/in.h>
#include <netinet/in_var.h>
#include <stdbool.h>

void
CGAInit(bool expire_ipv6ll_modifiers);

bool
CGAIsEnabled(void);

void
CGAPrepareSetForInterfaceLinkLocal(const char * name,
				   struct in6_cga_prepare * cgaprep);

void
CGAPrepareSetForInterface(const char * name, struct in6_cga_prepare * cgaprep);

#endif /* _S_CGA_H */
