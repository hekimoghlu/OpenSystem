/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 3, 2023.
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
 * This file is included by programs which are optionally built into the
 * shell.  If SHELL is defined, we try to map the standard UNIX library
 * routines to ash routines using defines.
 */

#include "../shell.h"
#include "../mystring.h"
#ifdef SHELL
#include "../error.h"
#include "../output.h"
#include "builtins.h"
#define FILE struct output
#undef stdout
#define stdout out1
#undef stderr
#define stderr out2
#define printf out1fmt
#undef putc
#define putc(c, file)	outc(c, file)
#undef putchar
#define putchar(c)	out1c(c)
#define fprintf outfmt
#define fputs outstr
#define fwrite(ptr, size, nmemb, file) outbin(ptr, (size) * (nmemb), file)
#define fflush flushout
#define INITARGS(argv)
#define warnx warning
#define warn(fmt, ...) warning(fmt ": %s", __VA_ARGS__, strerror(errno))
#define errx(exitstatus, ...) error(__VA_ARGS__)

#else
#undef NULL
#include <stdio.h>
#undef main
#define INITARGS(argv)	if ((commandname = argv[0]) == NULL) {fputs("Argc is zero\n", stderr); exit(2);} else
#endif

#include <unistd.h>

pointer stalloc(int);
int killjob(const char *, int);

extern char *commandname;
