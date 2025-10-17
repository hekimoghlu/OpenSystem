/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 30, 2023.
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
#include "codepage.h"

#ifdef WIN32
#include <windows.h>

int codepageMap(int cp, int *map)
{
  int i;
  CPINFO info;
  if (!GetCPInfo(cp, &info) || info.MaxCharSize > 2)
    return 0;
  for (i = 0; i < 256; i++)
    map[i] = -1;
  if (info.MaxCharSize > 1) {
    for (i = 0; i < MAX_LEADBYTES; i++) {
      int j, lim;
      if (info.LeadByte[i] == 0 && info.LeadByte[i + 1] == 0)
        break;
      lim = info.LeadByte[i + 1];
      for (j = info.LeadByte[i]; j < lim; j++)
	map[j] = -2;
    }
  }
  for (i = 0; i < 256; i++) {
   if (map[i] == -1) {
     char c = i;
     unsigned short n;
     if (MultiByteToWideChar(cp, MB_PRECOMPOSED|MB_ERR_INVALID_CHARS,
		             &c, 1, &n, 1) == 1)
       map[i] = n;
   }
  }
  return 1;
}

int codepageConvert(int cp, const char *p)
{
  unsigned short c;
  if (MultiByteToWideChar(cp, MB_PRECOMPOSED|MB_ERR_INVALID_CHARS,
		          p, 2, &c, 1) == 1)
    return c;
  return -1;
}

#else /* not WIN32 */

int codepageMap(int cp, int *map)
{
  return 0;
}

int codepageConvert(int cp, const char *p)
{
  return -1;
}

#endif /* not WIN32 */
