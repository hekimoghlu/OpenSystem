/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 20, 2024.
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
#ifndef _BANK_BANK_TYPES_H_
#define _BANK_BANK_TYPES_H_

#include <os/base.h>
#include <stdint.h>
#include <mach/mach_types.h>

#define MACH_VOUCHER_ATTR_BANK_NULL             ((mach_voucher_attr_recipe_command_t)601)
#define MACH_VOUCHER_ATTR_BANK_CREATE           ((mach_voucher_attr_recipe_command_t)610)
#define MACH_VOUCHER_ATTR_BANK_MODIFY_PERSONA   ((mach_voucher_attr_recipe_command_t)611)

#define MACH_VOUCHER_BANK_CONTENT_SIZE (500)

typedef uint32_t bank_action_t;
#define BANK_ORIGINATOR_PID           0x1
#define BANK_PERSONA_TOKEN            0x2
#define BANK_PERSONA_ID               0x3
#define BANK_PERSONA_ADOPT_ANY        0x4
#define BANK_ORIGINATOR_PROXIMATE_PID 0x5


#define PROC_PERSONA_INFO_FLAG_ADOPTION_ALLOWED 0x1

struct proc_persona_info {
	uint64_t unique_pid;
	int32_t  pid;
	uint32_t flags;
	uint32_t pidversion;
	uint32_t persona_id;
	uint32_t uid;
	uint32_t gid;
	uint8_t  macho_uuid[16];
};

struct persona_token {
	struct proc_persona_info originator;
	struct proc_persona_info proximate;
};

struct persona_modify_info {
	uint32_t persona_id;
	uint64_t unique_pid;
};

#ifdef PRIVATE
/* Redeem bank voucher on behalf of another process while changing the persona */
#define ENTITLEMENT_PERSONA_MODIFY    "com.apple.private.persona.modify"
#define ENTITLEMENT_PERSONA_NO_PROPAGATE "com.apple.private.personas.no.propagate"
/* Allow to adopt any persona when spawned in no-persona */
#define ENTITLEMENT_PERSONA_ADOPT_ANY    "com.apple.private.persona.adopt.any"
#endif /* PRIVATE */

#endif /* _BANK_BANK_TYPES_H_ */
