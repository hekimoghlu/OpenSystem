/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 22, 2022.
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
/* CL interface program to curl cli tool. */

#include <stdio.h>
#include <stdlib.h>

#include <milib.h>
#include <miptrnam.h>
#include <mih/callpgmv.h>

#ifndef CURLPGM
#define CURLPGM "CURL"
#endif

/* Variable-length string, with 16-bit length. */
struct vary2 {
  short len;
  char  string[5000];
};

/* Arguments from CL command. */
struct arguments {
  char         *pgm;            /* Program name. */
  struct vary2 *cmdargs;        /* Command line arguments. */
};

static int
is_ifs(char c)
{
  return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

static int
parse_command_line(const char *cmdargs, size_t len,
                   size_t *argc, char **argv,
                   size_t *argsize, char *argbuf)
{
  const char *endline = cmdargs + len;
  char quote = '\0';
  int inarg = 0;

  *argc = 0;
  *argsize = 0;

  while(cmdargs < endline) {
    char c = *cmdargs++;

    if(!inarg) {
      /* Skip argument separator. */
      if(is_ifs(c))
        continue;

      /* Start a new argument. */
      ++*argc;
      if(argv)
        *argv++ = argbuf;
      inarg = 1;
    }

    /* Check for quoting end. */
    if(quote && quote == c) {
      quote = '\0';
      continue;
    }

    /* Check for backslash-escaping. */
    if(quote != '\'' && c == '\\') {
      if(cmdargs >= endline) {
        fputs("Trailing backslash in command\n", stderr);
        return -1;
      }
      c = *cmdargs++;
    }
    else if(!quote && is_ifs(c)) {      /* Check for end of argument. */
      inarg = 0;
      c = '\0';         /* Will store a string terminator. */
    }

    /* Store argument character and count it. */
    if(argbuf)
      *argbuf++ = c;
    ++*argsize;
  }

  if(quote) {
    fprintf(stderr, "Unterminated quote: %c\n", quote);
    return -1;
  }

  /* Terminate last argument. */
  if(inarg) {
    if(argbuf)
      *argbuf = '\0';
    ++*argsize;
  }

  /* Terminate argument list. */
  if(argv)
    *argv = NULL;

  return 0;
}


int
main(int argsc, struct arguments *args)
{
  size_t argc;
  char **argv;
  size_t argsize;
  int i;
  int exitcode;
  char library[11];

  /* Extract current program library name. */
  for(i = 0; i < 10; i++) {
    char c = args->pgm[i];

    if(!c || c == '/')
      break;

    library[i] = c;
  }
  library[i] = '\0';

  /* Measure arguments size. */
  exitcode = parse_command_line(args->cmdargs->string, args->cmdargs->len,
                                &argc, NULL, &argsize, NULL);

  if(!exitcode) {
    /* Allocate space for parsed arguments. */
    argv = (char **) malloc((argc + 1) * sizeof(*argv) + argsize);
    if(!argv) {
      fputs("Memory allocation error\n", stderr);
      exitcode = -2;
    }
    else {
      _SYSPTR pgmptr = rslvsp(WLI_PGM, (char *) CURLPGM, library, _AUTH_NONE);
      _LU_Work_Area_T *luwrka = (_LU_Work_Area_T *) _LUWRKA();

      parse_command_line(args->cmdargs->string, args->cmdargs->len,
                         &argc, argv, &argsize, (char *) (argv + argc + 1));

      /* Call program. */
      _CALLPGMV((void *) &pgmptr, argv, argc);
      exitcode = luwrka->LU_RC;

      free(argv);
    }
  }

  return exitcode;
}
