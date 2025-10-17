/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 21, 2024.
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
char cvsroot_getopt_c[] = "$Id: getopt.c 10926 2008-11-11 22:17:40Z wsfulton $";

#include "swig.h"

static char **args;
static int numargs;
static int *marked;

/* -----------------------------------------------------------------------------
 * Swig_init_args()
 * 
 * Initialize the argument list handler.
 * ----------------------------------------------------------------------------- */

void Swig_init_args(int argc, char **argv) {
  int i;
  assert(argc > 0);
  assert(argv);

  numargs = argc;
  args = argv;
  marked = (int *) malloc(numargs * sizeof(int));
  for (i = 0; i < argc; i++) {
    marked[i] = 0;
  }
  marked[0] = 1;
}

/* -----------------------------------------------------------------------------
 * Swig_mark_arg()
 * 
 * Marks an argument as being parsed.
 * ----------------------------------------------------------------------------- */

void Swig_mark_arg(int n) {
  assert(marked);
  assert((n >= 0) && (n < numargs));
  marked[n] = 1;
}

/* -----------------------------------------------------------------------------
 * Swig_check_marked()
 *
 * Checks to see if argument has been picked up.
 * ----------------------------------------------------------------------------- */

int Swig_check_marked(int n) {
  assert((n >= 0) && (n < numargs));
  return marked[n];
}

/* -----------------------------------------------------------------------------
 * Swig_check_options()
 * 
 * Checkers for unprocessed command line options and errors.
 * ----------------------------------------------------------------------------- */

void Swig_check_options(int check_input) {
  int error = 0;
  int i;
  int max = check_input ? numargs - 1 : numargs;
  assert(marked);
  for (i = 1; i < max; i++) {
    if (!marked[i]) {
      Printf(stderr, "swig error : Unrecognized option %s\n", args[i]);
      error = 1;
    }
  }
  if (error) {
    Printf(stderr, "Use 'swig -help' for available options.\n");
    exit(1);
  }
  if (check_input && marked[numargs - 1]) {
    Printf(stderr, "Must specify an input file. Use -help for available options.\n");
    exit(1);
  }
}

/* -----------------------------------------------------------------------------
 * Swig_arg_error()
 * 
 * Generates a generic error message and exits.
 * ----------------------------------------------------------------------------- */

void Swig_arg_error(void) {
  Printf(stderr, "SWIG : Unable to parse command line options.\n");
  Printf(stderr, "Use 'swig -help' for available options.\n");
  exit(1);
}
