/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 30, 2021.
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

#include <stdio.h>
#include <errno.h>
#include "builtins.h"
#include "shell.h"

#if !defined (errno)
extern int errno;
#endif

rmdir_builtin (list)
     WORD_LIST *list;
{
  int rval;
  WORD_LIST *l;

  if (no_options (list))
    return (EX_USAGE);

  for (rval = EXECUTION_SUCCESS, l = list; l; l = l->next)
    if (rmdir (l->word->word) < 0)
      {
	builtin_error ("%s: %s", l->word->word, strerror (errno));
	rval = EXECUTION_FAILURE;
      }

  return rval;
}

char *rmdir_doc[] = {
	"rmdir removes the directory entry specified by each argument,",
	"provided the directory is empty.",
	(char *)NULL
};

/* The standard structure describing a builtin command.  bash keeps an array
   of these structures. */
struct builtin rmdir_struct = {
	"rmdir",		/* builtin name */
	rmdir_builtin,		/* function implementing the builtin */
	BUILTIN_ENABLED,	/* initial flags for builtin */
	rmdir_doc,		/* array of long documentation strings. */
	"rmdir directory ...",	/* usage synopsis; becomes short_doc */
	0			/* reserved for internal use */
};
