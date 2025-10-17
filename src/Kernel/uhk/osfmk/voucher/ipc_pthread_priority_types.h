/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 19, 2022.
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
#ifndef _VOUCHER_IPC_PTHREAD_PRIORITY_TYPES_H_
#define _VOUCHER_IPC_PTHREAD_PRIORITY_TYPES_H_

#include <stdint.h>
#include <mach/mach_types.h>

#define MACH_VOUCHER_ATTR_PTHPRIORITY_NULL              ((mach_voucher_attr_recipe_command_t)701)
#define MACH_VOUCHER_ATTR_PTHPRIORITY_CREATE            ((mach_voucher_attr_recipe_command_t)710)

typedef uint32_t ipc_pthread_priority_value_t;

#define MACH_VOUCHER_PTHPRIORITY_CONTENT_SIZE (sizeof(ipc_pthread_priority_value_t))

#endif /* _VOUCHER_IPC_PTHREAD_PRIORITY_TYPES_H_ */
