/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 1, 2022.
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
#ifndef _KERN_SPL_H_
#define _KERN_SPL_H_

#include <machine/machine_routines.h>

typedef unsigned spl_t;

#define splhigh()       (spl_t) ml_set_interrupts_enabled(FALSE)
#define splsched()      (spl_t) ml_set_interrupts_enabled(FALSE)
#define splclock()      (spl_t) ml_set_interrupts_enabled(FALSE)
#define splx(x)         (void) ml_set_interrupts_enabled(x)
#define spllo()         (void) ml_set_interrupts_enabled(TRUE)

#endif  /* _KERN_SPL_H_ */
