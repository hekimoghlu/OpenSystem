/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 2, 2024.
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
#if CONFIG_EXCLAVES

#pragma once

#include <mach/exclaves.h>

#include <libkern/section_keywords.h>
#include <mach/kern_return.h>

#define EXCLAVES_BOOT_TASK_SEGMENT "__DATA_CONST"
#define EXCLAVES_BOOT_TASK_SECTION "__exclaves_bt"

__BEGIN_DECLS

__enum_decl(exclaves_boot_task_rank_t, uint32_t, {
	EXCLAVES_BOOT_RANK_FIRST          = 0,
	EXCLAVES_BOOT_RANK_SECOND         = 1,
	EXCLAVES_BOOT_RANK_THIRD          = 2,
	EXCLAVES_BOOT_RANK_FOURTH         = 3,

	EXCLAVES_BOOT_RANK_ANY            = 0x7fffffff,

	EXCLAVES_BOOT_RANK_LAST           = 0xffffffff,
});

typedef struct exclaves_boot_task_entry {
	kern_return_t (*ebt_func)(void);
	exclaves_boot_task_rank_t ebt_rank;
	const char *ebt_name;
} exclaves_boot_task_entry_t;

/* BEGIN IGNORE CODESTYLE */
#define __EXCLAVES_BOOT_TASK(name, line, rank, func)              \
	__PLACE_IN_SECTION(EXCLAVES_BOOT_TASK_SEGMENT ","         \
	    EXCLAVES_BOOT_TASK_SECTION)                           \
	static const exclaves_boot_task_entry_t                   \
	__exclaves_boot_task_entry_ ## name ## _ ## line = {      \
	    .ebt_func = func,                                     \
	    .ebt_rank = rank,                                     \
	    /* Used for  panic string. */                         \
	    .ebt_name = #name,                                    \
	}
/* END IGNORE CODESTYLE */

#define EXCLAVES_BOOT_TASK(func, rank)                            \
	__EXCLAVES_BOOT_TASK(func, __LINE__, rank, func)

/* Returns the exclaves boot status as a string, for panic log reporting. */
extern const char *exclaves_get_boot_status_string(void);

/* Boot the requested boot stage. */
extern kern_return_t exclaves_boot(exclaves_boot_stage_t);

__END_DECLS

#endif /* CONFIG_EXCLAVES */
