/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 8, 2022.
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
#ifndef _FSLOG_H_
#define _FSLOG_H_

#include <sys/syslog.h>
#include <sys/mount.h>
#include <machine/limits.h>

#ifdef XNU_KERNEL_PRIVATE
/* Log information about external modification of a target process */
void fslog_extmod_msgtracer(proc_t caller, proc_t target);
#endif /* XNU_KERNEL_PRIVATE */

/* Keys used by FSLog */
#define FSLOG_KEY_LEVEL         "Level"         /* Priority level */

/* Values used by FSLog */
#define FSLOG_VAL_FACILITY      "com.apple.system.fs" /* Facility generating messages */

#endif /* !_FSLOG_H_ */
