/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 29, 2022.
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
 * Mach Operating System
 * Copyright (c) 1991,1990 Carnegie Mellon University
 * All Rights Reserved.
 *
 * Permission to use, copy, modify and distribute this software and its
 * documentation is hereby granted, provided that both the copyright
 * notice and this permission notice appear in all copies of the
 * software, derivative works or modified versions, and any portions
 * thereof, and that both notices appear in supporting documentation.
 *
 * CARNEGIE MELLON ALLOWS FREE USE OF THIS SOFTWARE IN ITS
 * CONDITION.  CARNEGIE MELLON DISCLAIMS ANY LIABILITY OF ANY KIND FOR
 * ANY DAMAGES WHATSOEVER RESULTING FROM THE USE OF THIS SOFTWARE.
 *
 * Carnegie Mellon requests users of this software to return to
 *
 *  Software Distribution Coordinator  or  Software.Distribution@CS.CMU.EDU
 *  School of Computer Science
 *  Carnegie Mellon University
 *  Pittsburgh PA 15213-3890
 *
 * any improvements or extensions that they make and grant Carnegie the
 * rights to redistribute these changes.
 */

#include <mach/boolean.h>
#include <ctype.h>
#include "error.h"
#include "alloc.h"
#include "strdefs.h"

string_t
strmake(char *string)
{
  string_t saved;
  
  saved = malloc(strlen(string) + 1);
  if (saved == strNULL)
    fatal("strmake('%s'): %s", string, strerror(errno));
  return strcpy(saved, string);
}

string_t
strconcat(string_t left, string_t right)
{
  string_t saved;
  
  saved = malloc(strlen(left) + strlen(right) + 1);
  if (saved == strNULL)
    fatal("strconcat('%s', '%s'): %s", left, right, strerror(errno));
  return strcat(strcpy(saved, left), right);
}

string_t
strphrase(string_t left, string_t right)
{
  string_t saved;
  string_t current;
  size_t llen;
  
  llen = strlen(left);
  saved = malloc(llen + strlen(right) + 2);
  if (saved == strNULL)
    fatal("strphrase('%s', '%s'): %s", left, right, strerror(errno));
  strcpy(saved, left);
  current = saved + llen;
  *(current++) = ' ';
  strcpy(current, right);
  free(left);
  return(saved);
}

void
strfree(string_t string)
{
  free(string);
}

char *
strbool(boolean_t bool)
{
  if (bool)
    return "TRUE";
  else
    return "FALSE";
}

char *
strstring(string_t string)
{
  if (string == strNULL)
    return "NULL";
  else
    return string;
}

char *
toupperstr(char *p)
{
  char *s = p;
  char c;
  
  while ((c = *s)) {
    if (islower(c))
      *s = toupper(c);
    s++;
  }
  return(p);
}
