/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 3, 2023.
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
/* Should only be used to remove directories by a superuser prepared to let
   fsck clean up the file system. */

#include <config.h>

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#include <stdio.h>
#include <errno.h>

#include "builtins.h"
#include "shell.h"

#ifndef errno
extern int errno;
#endif

unlink_builtin (list)
     WORD_LIST *list;
{
  if (list == 0)
    {
      builtin_usage ();
      return (EX_USAGE);
    }

  if (unlink (list->word->word) != 0)
    {
      builtin_error ("%s: cannot unlink: %s", list->word->word, strerror (errno));
      return (EXECUTION_FAILURE);
    }

  return (EXECUTION_SUCCESS);
}

char *unlink_doc[] = {
	"Remove a directory entry.",
	(char *)NULL
};

struct builtin unlink_struct = {
	"unlink",		/* builtin name */
	unlink_builtin,		/* function implementing the builtin */
	BUILTIN_ENABLED,	/* initial flags for builtin */
	unlink_doc,		/* array of long documentation strings. */
	"unlink name",		/* usage synopsis; becomes short_doc */
	0			/* reserved for internal use */
};
