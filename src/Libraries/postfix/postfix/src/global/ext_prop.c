/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 3, 2025.
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
/* System library. */

#include <sys_defs.h>

/* Utility library. */

#include <name_mask.h>

/* Global library. */

#include <mail_params.h>
#include <ext_prop.h>

/* ext_prop_mask - compute extension propagation mask */

int     ext_prop_mask(const char *param_name, const char *pattern)
{
    static const NAME_MASK table[] = {
	"canonical", EXT_PROP_CANONICAL,
	"virtual", EXT_PROP_VIRTUAL,
	"alias", EXT_PROP_ALIAS,
	"forward", EXT_PROP_FORWARD,
	"include", EXT_PROP_INCLUDE,
	"generic", EXT_PROP_GENERIC,
	0,
    };

    return (name_mask(param_name, table, pattern));
}
