/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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
 * @OSF_COPYRIGHT@
 */


#ifndef _MACH_DEFAULT_PAGER_TYPES_H_
#define _MACH_DEFAULT_PAGER_TYPES_H_

#include <sys/appleapiopts.h>

#ifdef __APPLE_API_UNSTABLE

#include <mach/mach_types.h>
#include <mach/machine/vm_types.h>
#include <mach/memory_object_types.h>

#define HI_WAT_ALERT            0x01
#define LO_WAT_ALERT            0x02
#define SWAP_ENCRYPT_ON         0x04
#define SWAP_ENCRYPT_OFF        0x08
#define SWAP_COMPACT_DISABLE    0x10
#define SWAP_COMPACT_ENABLE     0x20
#define PROC_RESUME             0x40
#define SWAP_FILE_CREATION_ERROR        0x80
#define USE_EMERGENCY_SWAP_FILE_FIRST   0x100

#endif /* __APPLE_API_UNSTABLE */

#endif  /* _MACH_DEFAULT_PAGER_TYPES_H_ */
