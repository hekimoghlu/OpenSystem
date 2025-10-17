/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 3, 2023.
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
#include "bashtypes.h"
#include "shell.h"
#include "builtins.h"

true_builtin (list)
     WORD_LIST *list;
{
  return EXECUTION_SUCCESS;
}

false_builtin (list)
     WORD_LIST *list;
{
  return EXECUTION_FAILURE;
}

static char *true_doc[] = {
	"Return a successful result.",
	(char *)NULL
};

static char *false_doc[] = {
	"Return an unsuccessful result.",
	(char *)NULL
};

struct builtin true_struct = {
	"true",
	true_builtin,
	BUILTIN_ENABLED,
	true_doc,
	"true",
	0
};

struct builtin false_struct = {
	"false",
	false_builtin,
	BUILTIN_ENABLED,
	false_doc,
	"false",
	0
};
