/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 25, 2024.
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
#ifndef _ATM_ATM_INTERNAL_H_
#define _ATM_ATM_INTERNAL_H_

#include <stdint.h>
#include <mach/mach_types.h>
#include <atm/atm_types.h>
#include <os/refcnt.h>

#ifdef MACH_KERNEL_PRIVATE
void atm_init(void);
#endif /* MACH_KERNEL_PRIVATE */

#ifdef XNU_KERNEL_PRIVATE
void atm_reset(void);
#endif /* XNU_KERNEL_PRIVATE */

kern_return_t atm_set_diagnostic_config(uint32_t);
uint32_t atm_get_diagnostic_config(void);

#endif /* _ATM_ATM_INTERNAL_H_ */
