/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 25, 2025.
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

#include <fcntl.h>
#include <errno.h>

#include "builtins.h"
#include "shell.h"

#ifndef errno
extern int errno;
#endif

extern char **make_builtin_argv ();
extern char **export_env;

extern int perl_main();

bperl_builtin(list)
WORD_LIST *list;
{
	char	**v;
	int	c, r;

	v = make_builtin_argv(list, &c);
	r = perl_main(c, v, export_env);
	free(v);

	return r;
}

char *bperl_doc[] = {
	"An interface to a perl5 interpreter.",
	(char *)0
};

struct builtin bperl_struct = {
	"bperl",
	bperl_builtin,
	BUILTIN_ENABLED,
	bperl_doc,
	"bperl [perl options] [file ...]",
	0
};
