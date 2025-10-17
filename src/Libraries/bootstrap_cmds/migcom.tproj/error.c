/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 22, 2025.
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

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>

#include "global.h"
#include "error.h"

extern int lineno;
extern char *yyinname;

static char *program;
__private_extern__
int mig_errors = 0;

/*ARGSUSED*/
/*VARARGS1*/
void
fatal(char *format, ...)
{
  va_list pvar;
  va_start(pvar, format);
  fprintf(stderr, "%s: fatal: \"%s\", line %d: ", program, yyinname, lineno-1);
  (void) vfprintf(stderr, format, pvar);
  fprintf(stderr, "\n");
  va_end(pvar);
  exit(1);
}

__private_extern__
/*ARGSUSED*/
/*VARARGS1*/
void
warn(char *format, ...)
{
  va_list pvar;
  va_start(pvar, format);
  if (!BeQuiet && (mig_errors == 0)) {
    fprintf(stderr, "\"%s\", line %d: warning: ", yyinname, lineno-1);
    (void) vfprintf(stderr, format, pvar);
    fprintf(stderr, "\n");
  }
  va_end(pvar);
}

/*ARGSUSED*/
/*VARARGS1*/
void
error(char *format, ...)
{
  va_list pvar;
  va_start(pvar, format);
  fprintf(stderr, "\"%s\", line %d: ", yyinname, lineno-1);
  (void) vfprintf(stderr, format, pvar);
  fprintf(stderr, "\n");
  va_end(pvar);
  mig_errors++;
}

void
set_program_name(char *name)
{
  program = name;
}
