/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 1, 2025.
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

dirname_builtin (list)
     WORD_LIST *list;
{
  int slen;
  char *string;

  if (list == 0 || list->next)
    {
      builtin_usage ();
      return (EX_USAGE);
    }

  if (no_options (list))
    return (EX_USAGE);

  string = list->word->word;
  slen = strlen (string);

  /* Strip trailing slashes */
  while (slen > 0 && string[slen - 1] == '/')
    slen--;

  /* (2) If string consists entirely of slash characters, string shall be
	 set to a single slash character.  In this case, skip steps (3)
	 through (8). */
  if (slen == 0)
    {
      fputs ("/\n", stdout);
      return (EXECUTION_SUCCESS);
    }

  /* (3) If there are any trailing slash characters in string, they
	 shall be removed. */
  string[slen] = '\0';

  /* (4) If there are no slash characters remaining in string, string
	 shall be set to a single period character.  In this case, skip
	 steps (5) through (8).

     (5) If there are any trailing nonslash characters in string,
	 they shall be removed. */

  while (--slen >= 0)
    if (string[slen] == '/')
      break;

  if (slen < 0)
    {
      fputs (".\n", stdout);
      return (EXECUTION_SUCCESS);
    }

  /* (7) If there are any trailing slash characters in string, they
	 shall be removed. */
  while (--slen >= 0)
    if (string[slen] != '/')
      break;
  string[++slen] = '\0';

  /* (8) If the remaining string is empty, string shall be set to a single
	 slash character. */
  printf ("%s\n", (slen == 0) ? "/" : string);
  return (EXECUTION_SUCCESS);
}

char *dirname_doc[] = {
	"The STRING is converted to the name of the directory containing",	
	"the filename corresponding to the last pathname component in STRING.",
	(char *)NULL
};

/* The standard structure describing a builtin command.  bash keeps an array
   of these structures. */
struct builtin dirname_struct = {
	"dirname",		/* builtin name */
	dirname_builtin,	/* function implementing the builtin */
	BUILTIN_ENABLED,	/* initial flags for builtin */
	dirname_doc,		/* array of long documentation strings. */
	"dirname string",	/* usage synopsis */
	0			/* reserved for internal use */
};
