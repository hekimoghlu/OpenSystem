/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 13, 2025.
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
#include <string.h>
#ifdef HAVE_STRINGS_H
# include <strings.h>
#endif /* HAVE_STRINGS_H */

#include "sudo_compat.h"
#include "sudo_debug.h"
#include "sudo_util.h"

int
sudo_strtobool_v1(const char *str)
{
    debug_decl(sudo_strtobool, SUDO_DEBUG_UTIL);

    switch (*str) {
	case '0':
	case '1':
	    if (str[1] == '\0')
		debug_return_int(*str - '0');
	    break;
	case 'y':
	case 'Y':
	    if (strcasecmp(str, "yes") == 0)
		debug_return_int(1);
	    break;
	case 't':
	case 'T':
	    if (strcasecmp(str, "true") == 0)
		debug_return_int(1);
	    break;
	case 'o':
	case 'O':
	    if (strcasecmp(str, "on") == 0)
		debug_return_int(1);
	    if (strcasecmp(str, "off") == 0)
		debug_return_int(0);
	    break;
	case 'n':
	case 'N':
	    if (strcasecmp(str, "no") == 0)
		debug_return_int(0);
	    break;
	case 'f':
	case 'F':
	    if (strcasecmp(str, "false") == 0)
		debug_return_int(0);
	    break;
    }
    sudo_debug_printf(SUDO_DEBUG_ERROR|SUDO_DEBUG_LINENO,
	"invalid boolean value \"%s\"", str);

    debug_return_int(-1);
}
