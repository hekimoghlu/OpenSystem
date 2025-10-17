/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 21, 2024.
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
 * Modification History
 *
 * November 8, 2001	Dieter Siegmund
 * - created
 */

#ifndef _S_PRINTDATA_H
#define _S_PRINTDATA_H

#include <sys/types.h>
#include <stdint.h>
#include <stdio.h>
#include <CoreFoundation/CFString.h>

void
fprint_data(FILE * f, const uint8_t * data, int len);

void
print_data(const uint8_t * data, int len);

void
fprint_bytes(FILE * out_f, const uint8_t * data_p, int n_bytes);

void
print_bytes(const uint8_t * data, int len);

void
print_bytes_cfstr(CFMutableStringRef str, const uint8_t * data_p,
		  int n_bytes);
void
print_data_cfstr(CFMutableStringRef str, const uint8_t * data_p,
		 int n_bytes);

#endif /* _S_PRINTDATA_H */

