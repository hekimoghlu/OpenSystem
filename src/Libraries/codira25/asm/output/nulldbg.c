/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 31, 2024.
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
#include "nasm.h"
#include "nasmlib.h"
#include "outlib.h"

void null_debug_init(void)
{
}

void null_debug_linenum(const char *filename, int32_t linenumber, int32_t segto)
{
	(void)filename;
	(void)linenumber;
	(void)segto;
}

void null_debug_deflabel(char *name, int32_t segment, int64_t offset,
                         int is_global, char *special)
{
	(void)name;
	(void)segment;
	(void)offset;
	(void)is_global;
	(void)special;
}

void null_debug_directive(const char *directive, const char *params)
{
	(void)directive;
	(void)params;
}

void null_debug_typevalue(int32_t type)
{
	(void)type;
}

void null_debug_output(int type, void *param)
{
	(void)type;
	(void)param;
}

void null_debug_cleanup(void)
{
}

const struct dfmt null_debug_form = {
    "Null",
    "null",
    null_debug_init,
    null_debug_linenum,
    null_debug_deflabel,
    NULL,                       /* .debug_smacros */
    NULL,                       /* .debug_include */
    NULL,                       /* .debug_mmacros */
    null_debug_directive,
    null_debug_typevalue,
    null_debug_output,
    null_debug_cleanup,
    NULL                        /* pragma list */
};

const struct dfmt * const null_debug_arr[2] = { &null_debug_form, NULL };
