/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 15, 2024.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "linenoise.h"

void completion(const char *buf, linenoiseCompletions *lc) {
  if (buf[0] == 'h') {
    linenoiseAddCompletion(lc, "hello");
    linenoiseAddCompletion(lc, "hello there");
  }
}

int main(int argc, char **argv) {
  char *line;
  char *prgname = argv[0];

  /* Parse options, with --multiline we enable multi line editing. */
  while (argc > 1) {
    argc--;
    argv++;
    if (!strcmp(*argv, "--multiline")) {
      linenoiseSetMultiLine(1);
      printf("Multi-line mode enabled.\n");
    } else if (!strcmp(*argv, "--keycodes")) {
      linenoisePrintKeyCodes();
      exit(0);
    } else {
      fprintf(stderr, "Usage: %s [--multiline] [--keycodes]\n", prgname);
      exit(1);
    }
  }

  /* Set the completion callback. This will be called every time the
   * user uses the <tab> key. */
  linenoiseSetCompletionCallback(completion);

  /* Load history from file. The history file is just a plain text file
   * where entries are separated by newlines. */
  linenoiseHistoryLoad("history.txt"); /* Load the history at startup */

  /* Now this is the main loop of the typical linenoise-based application.
   * The call to linenoise() will block as long as the user types something
   * and presses enter.
   *
   * The typed string is returned as a malloc() allocated string by
   * linenoise, so the user needs to free() it. */
  while ((line = linenoise("hello> ")) != NULL) {
    /* Do something with the string. */
    if (line[0] != '\0' && line[0] != '/') {
      printf("echo: '%s'\n", line);
      linenoiseHistoryAdd(line);           /* Add to the history. */
      linenoiseHistorySave("history.txt"); /* Save the history on disk. */
    } else if (!strncmp(line, "/historylen", 11)) {
      /* The "/historylen" command will change the history len. */
      int len = atoi(line + 11);
      linenoiseHistorySetMaxLen(len);
    } else if (line[0] == '/') {
      printf("Unreconized command: %s\n", line);
    }
    free(line);
  }
  return 0;
}
