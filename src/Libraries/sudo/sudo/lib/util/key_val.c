/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 21, 2023.
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
 * This is an open source non-commercial project. Dear PVS-Studio, please check it.
 * PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
 */

#include <config.h>

#include <stdlib.h>
#include <string.h>

#include "sudo_compat.h"
#include "sudo_debug.h"
#include "sudo_util.h"

/*
 * Create a new key=value pair and return it.
 * The caller is responsible for freeing the string.
 */
char *
sudo_new_key_val_v1(const char *key, const char *val)
{
    size_t key_len = strlen(key);
    size_t val_len = strlen(val);
    char *cp, *str;
    debug_decl(sudo_new_key_val, SUDO_DEBUG_UTIL);

    cp = str = malloc(key_len + 1 + val_len + 1);
    if (cp != NULL) {
	memcpy(cp, key, key_len);
	cp += key_len;
	*cp++ = '=';
	memcpy(cp, val, val_len);
	cp += val_len;
	*cp = '\0';
    }

    debug_return_str(str);
}
