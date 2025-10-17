/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 28, 2022.
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
/* Sample builtin to be dynamically loaded with enable -f and replace an
   existing builtin. */

#include <stdio.h>
#include "builtins.h"
#include "shell.h"

necho_builtin (list)
WORD_LIST *list;
{
	print_word_list (list, " ");
	printf("\n");
	fflush (stdout);
	return (EXECUTION_SUCCESS);
}

char *necho_doc[] = {
	"Print the arguments to the standard ouput separated",
	"by space characters and terminated with a newline.",
	(char *)NULL
};
	
struct builtin necho_struct = {
	"echo",
	necho_builtin,
	BUILTIN_ENABLED,
	necho_doc,
	"echo [args]",
	0
};
	
