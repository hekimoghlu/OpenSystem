/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 19, 2025.
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
/* See Makefile for compilation details. */

#include "config.h"

#if defined (HAVE_UNISTD_H)
#  include <unistd.h>
#endif

#include <stdio.h>
#include "builtins.h"
#include "shell.h"

basename_builtin (list)
     WORD_LIST *list;
{
  int slen, sufflen, off;
  char *string, *suffix, *fn;

  if (list == 0)
    {
      builtin_usage ();
      return (EX_USAGE);
    }

  if (no_options (list))
    return (EX_USAGE);

  string = list->word->word;
  suffix = (char *)NULL;
  if (list->next)
    {
      list = list->next;
      suffix = list->word->word;
    }

  if (list->next)
    {
      builtin_usage ();
      return (EX_USAGE);
    }

  slen = strlen (string);

  /* Strip trailing slashes */
  while (slen > 0 && string[slen - 1] == '/')
    slen--;

  /* (2) If string consists entirely of slash characters, string shall be
	 set to a single slash character.  In this case, skip steps (3)
	 through (5). */
  if (slen == 0)
    {
      fputs ("/\n", stdout);
      return (EXECUTION_SUCCESS);
    }

  /* (3) If there are any trailing slash characters in string, they
	 shall be removed. */
  string[slen] = '\0';

  /* (4) If there are any slash characters remaining in string, the prefix
	 of string up to an including the last slash character in string
	 shall be removed. */
  while (--slen >= 0)
    if (string[slen] == '/')
      break;

  fn = string + slen + 1;

  /* (5) If the suffix operand is present, is not identical to the
	 characters remaining in string, and is identical to a suffix
	 of the characters remaining in string, the suffix suffix
	 shall be removed from string.  Otherwise, string shall not be
	 modified by this step. */
  if (suffix)
    {
      sufflen = strlen (suffix);
      slen = strlen (fn);
      if (sufflen < slen)
        {
          off = slen - sufflen;
          if (strcmp (fn + off, suffix) == 0)
            fn[off] = '\0';
        }
    }
  printf ("%s\n", fn);
  return (EXECUTION_SUCCESS);
}

char *basename_doc[] = {
	"The STRING is converted to a filename corresponding to the last",
	"pathname component in STRING.  If the suffix string SUFFIX is",
	"supplied, it is removed.",
	(char *)NULL
};

/* The standard structure describing a builtin command.  bash keeps an array
   of these structures. */
struct builtin basename_struct = {
	"basename",		/* builtin name */
	basename_builtin,	/* function implementing the builtin */
	BUILTIN_ENABLED,	/* initial flags for builtin */
	basename_doc,		/* array of long documentation strings. */
	"basename string [suffix]",	/* usage synopsis */
	0			/* reserved for internal use */
};
