/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 20, 2024.
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
#ifndef _ATM_ATM_TYPES_H_
#define _ATM_ATM_TYPES_H_

#include <stdint.h>
#include <mach/mach_types.h>

/* Everything here is Deprecated. will be removed soon */

#define MACH_VOUCHER_ATTR_ATM_NULL              ((mach_voucher_attr_recipe_command_t)501)
#define MACH_VOUCHER_ATTR_ATM_CREATE            ((mach_voucher_attr_recipe_command_t)510)
#define MACH_VOUCHER_ATTR_ATM_REGISTER          ((mach_voucher_attr_recipe_command_t)511)

typedef uint32_t atm_action_t;
#define ATM_ACTION_DISCARD      0x1
#define ATM_ACTION_COLLECT      0x2
#define ATM_ACTION_LOGFAIL      0x3
#define ATM_FIND_MIN_SUB_AID    0x4
#define ATM_ACTION_UNREGISTER   0x5
#define ATM_ACTION_REGISTER     0x6
#define ATM_ACTION_GETSUBAID    0x7

typedef uint64_t atm_guard_t;
typedef uint64_t aid_t;
typedef uint64_t subaid_t;
typedef uint64_t mailbox_offset_t;
#define SUB_AID_MAX (UINT64_MAX)

typedef uint64_t atm_aid_t;
typedef uint32_t atm_subaid32_t;
typedef uint64_t mach_atm_subaid_t;             /* Used for mach based apis. */
typedef uint64_t atm_mailbox_offset_t;

typedef mach_port_t atm_memory_descriptor_t;
typedef atm_memory_descriptor_t *atm_memory_descriptor_array_t;
typedef uint64_t *atm_memory_size_array_t;

#define ATM_SUBAID32_MAX                (UINT32_MAX)
#define ATM_TRACE_DISABLE               (0x0100) /* OS_TRACE_MODE_DISABLE - Do not initialize the new logging*/
#define ATM_TRACE_OFF                   (0x0400) /* OS_TRACE_MODE_OFF - Don't drop log messages to new log buffers */
#define ATM_ENABLE_LEGACY_LOGGING       (0x20000000) /* OS_TRACE_SYSTEMMODE_LEGACY_LOGGING - Enable legacy logging  */

#endif /* _ATM_ATM_TYPES_H_ */
