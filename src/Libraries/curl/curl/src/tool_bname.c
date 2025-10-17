/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 20, 2021.
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
#include "tool_setup.h"

#include "tool_bname.h"

#include "memdebug.h" /* keep this as LAST include */

#ifndef HAVE_BASENAME

char *tool_basename(char *path)
{
  char *s1;
  char *s2;

  s1 = strrchr(path, '/');
  s2 = strrchr(path, '\\');

  if(s1 && s2) {
    path = (s1 > s2) ? s1 + 1 : s2 + 1;
  }
  else if(s1)
    path = s1 + 1;
  else if(s2)
    path = s2 + 1;

  return path;
}

#endif /* HAVE_BASENAME */
