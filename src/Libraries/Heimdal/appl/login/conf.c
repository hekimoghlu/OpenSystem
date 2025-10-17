/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 20, 2024.
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
#include "login_locl.h"

RCSID("$Id$");

static char *confbuf;

static int
login_conf_init(void)
{
    char *files[] = { _PATH_LOGIN_CONF, NULL };
    return cgetent(&confbuf, files, "default");
}

char *
login_conf_get_string(const char *str)
{
    char *value;
    if(login_conf_init() != 0)
	return NULL;
    if(cgetstr(confbuf, (char *)str, &value) < 0)
	return NULL;
    return value;
}
