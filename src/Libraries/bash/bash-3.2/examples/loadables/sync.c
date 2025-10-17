/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 9, 2024.
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

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#include "builtins.h"
#include "shell.h"
#include "bashgetopt.h"

sync_builtin (list)
     WORD_LIST *list;
{
  sync();
  return (EXECUTION_SUCCESS);
}

char *sync_doc[] = {
	"force completion of pending disk writes",
	(char *)NULL
};

struct builtin sync_struct = {
	"sync",			/* builtin name */
	sync_builtin,		/* function implementing the builtin */
	BUILTIN_ENABLED,	/* initial flags for builtin */
	sync_doc,		/* array of long documentation strings. */
	"sync",			/* usage synopsis; becomes short_doc */
	0			/* reserved for internal use */
};
