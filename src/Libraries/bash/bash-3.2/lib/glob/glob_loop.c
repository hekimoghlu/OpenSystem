/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 14, 2024.
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
static int INTERNAL_GLOB_PATTERN_P __P((const CHAR *));

/* Return nonzero if PATTERN has any special globbing chars in it.
   Compiled twice, once each for single-byte and multibyte characters. */
static int
INTERNAL_GLOB_PATTERN_P (pattern)
     const CHAR *pattern;
{
  register const CHAR *p;
  register CHAR c;
  int bopen;

  p = pattern;
  bopen = 0;

  while ((c = *p++) != L('\0'))
    switch (c)
      {
      case L('?'):
      case L('*'):
	return 1;

      case L('['):      /* Only accept an open brace if there is a close */
	bopen++;        /* brace to match it.  Bracket expressions must be */
	continue;       /* complete, according to Posix.2 */
      case L(']'):
	if (bopen)
	  return 1;
	continue;

      case L('+'):         /* extended matching operators */
      case L('@'):
      case L('!'):
	if (*p == L('('))  /*) */
	  return 1;
	continue;

      case L('\\'):
	if (*p++ == L('\0'))
	  return 0;
      }

  return 0;
}

#undef INTERNAL_GLOB_PATTERN_P
#undef L
#undef INT
#undef CHAR
