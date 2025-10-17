/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 7, 2022.
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
#ifndef _KERN_LOCKDOWN_MODE_H_
#define _KERN_LOCKDOWN_MODE_H_

#include <sys/cdefs.h>
__BEGIN_DECLS

#if KERNEL_PRIVATE
/* XNU and Kernel extensions */

#if XNU_KERNEL_PRIVATE

// Whether Lockdown Mode is enabled via the "ldm" nvram variable
extern int lockdown_mode_state;

/**
 * Initalizes Lockdown Mode
 */
void lockdown_mode_init(void);

#endif /* XNU_KERNEL_PRIVATE */

/**
 * Returns the Lockdown Mode enablement state
 */
int get_lockdown_mode_state(void);

/**
 * Enable Lockdown Mode by setting the nvram variable
 */
void enable_lockdown_mode(void);

/**
 * Disable Lockdown Mode by setting the nvram variable
 */
void disable_lockdown_mode(void);

#endif /* KERNEL_PRIVATE */
__END_DECLS

#endif /* _KERN_LOCKDOWN_MODE_H_ */
