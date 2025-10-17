/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 29, 2023.
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
#include <config.h>

#include <stdlib.h>
#include <string.h>

#include "roken.h"

#if !HAVE_DECL_ENVIRON
extern char **environ;
#endif

/*
 * unsetenv --
 */
ROKEN_LIB_FUNCTION void ROKEN_LIB_CALL
unsetenv(const char *name)
{
  int len;
  const char *np;
  char **p;

  if (name == 0 || environ == 0)
    return;

  for (np = name; *np && *np != '='; np++)
    /* nop */;
  len = np - name;

  for (p = environ; *p != 0; p++)
    if (strncmp(*p, name, len) == 0 && (*p)[len] == '=')
      break;

  for (; *p != 0; p++)
    *p = *(p + 1);
}

