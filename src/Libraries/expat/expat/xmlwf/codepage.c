/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 21, 2024.
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
#include "internal.h" /* for UNUSED_P only */

#if defined(_WIN32)
#  define STRICT 1
#  define WIN32_LEAN_AND_MEAN 1

#  include <windows.h>
#endif /* defined(_WIN32) */

int
codepageMap(int cp, int *map) {
#if defined(_WIN32)
  int i;
  CPINFO info;
  if (! GetCPInfo(cp, &info) || info.MaxCharSize > 2)
    return 0;
  for (i = 0; i < 256; i++)
    map[i] = -1;
  if (info.MaxCharSize > 1) {
    for (i = 0; i < MAX_LEADBYTES; i += 2) {
      int j, lim;
      if (info.LeadByte[i] == 0 && info.LeadByte[i + 1] == 0)
        break;
      lim = info.LeadByte[i + 1];
      for (j = info.LeadByte[i]; j <= lim; j++)
        map[j] = -2;
    }
  }
  for (i = 0; i < 256; i++) {
    if (map[i] == -1) {
      char c = (char)i;
      unsigned short n;
      if (MultiByteToWideChar(cp, MB_PRECOMPOSED | MB_ERR_INVALID_CHARS, &c, 1,
                              &n, 1)
          == 1)
        map[i] = n;
    }
  }
  return 1;
#else
  UNUSED_P(cp);
  UNUSED_P(map);
  return 0;
#endif
}

int
codepageConvert(int cp, const char *p) {
#if defined(_WIN32)
  unsigned short c;
  if (MultiByteToWideChar(cp, MB_PRECOMPOSED | MB_ERR_INVALID_CHARS, p, 2, &c,
                          1)
      == 1)
    return c;
  return -1;
#else
  UNUSED_P(cp);
  UNUSED_P(p);
  return -1;
#endif
}

