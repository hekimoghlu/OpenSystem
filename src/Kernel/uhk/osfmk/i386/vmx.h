/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 7, 2021.
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
#ifndef _I386_VMX_H_
#define _I386_VMX_H_

#if defined(__cplusplus)
extern "C" {
#endif

#include <mach/boolean.h>

/*
 * Error codes
 */
#define VMX_OK                  0 /* all ok */
#define VMX_UNSUPPORTED 1 /* VT unsupported or disabled on 1+ cores */
#define VMX_INUSE               2 /* VT is being exclusively used already */

/* SPI */
int host_vmxon(boolean_t exclusive);
void host_vmxoff(void);

#if defined(__cplusplus)
}
#endif

#endif
