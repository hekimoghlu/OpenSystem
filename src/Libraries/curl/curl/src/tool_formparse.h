/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 19, 2023.
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

#ifndef HEADER_CURL_TOOL_FORMPARSE_H
#define HEADER_CURL_TOOL_FORMPARSE_H
/***************************************************************************
 *                                  _   _ ____  _
 *  Project                     ___| | | |  _ \| |
 *                             / __| | | | |_) | |
 *                            | (__| |_| |  _ <| |___
 *                             \___|\___/|_| \_\_____|
 *
 * Copyright (C) Daniel Stenberg, <daniel@haxx.se>, et al.
 *
 * This software is licensed as described in the file COPYING, which
 * you should have received as part of this distribution. The terms
 * are also available at https://curl.se/docs/copyright.html.
 *
 * You may opt to use, copy, modify, merge, publish, distribute and/or sell
 * copies of the Software, and permit persons to whom the Software is
 * furnished to do so, under the terms of the COPYING file.
 *
 * This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
 * KIND, either express or implied.
 *
 * SPDX-License-Identifier: curl
 *
 ***************************************************************************/
#include "tool_setup.h"

/* Private structure for mime/parts. */

typedef enum {
  TOOLMIME_NONE = 0,
  TOOLMIME_PARTS,
  TOOLMIME_DATA,
  TOOLMIME_FILE,
  TOOLMIME_FILEDATA,
  TOOLMIME_STDIN,
  TOOLMIME_STDINDATA
} toolmimekind;

struct tool_mime {
  /* Structural fields. */
  toolmimekind kind;            /* Part kind. */
  struct tool_mime *parent;     /* Parent item. */
  struct tool_mime *prev;       /* Previous sibling (reverse order link). */
  /* Common fields. */
  char *data;                   /* Actual data or data filename. */
  char *name;                   /* Part name. */
  char *filename;               /* Part's filename. */
  char *type;                   /* Part's mime type. */
  char *encoder;                /* Part's requested encoding. */
  struct curl_slist *headers;   /* User-defined headers. */
  /* TOOLMIME_PARTS fields. */
  struct tool_mime *subparts;   /* Part's subparts. */
  /* TOOLMIME_STDIN/TOOLMIME_STDINDATA fields. */
  curl_off_t origin;            /* Stdin read origin offset. */
  curl_off_t size;              /* Stdin data size. */
  curl_off_t curpos;            /* Stdin current read position. */
  struct GlobalConfig *config;  /* For access from callback. */
};

size_t tool_mime_stdin_read(char *buffer,
                            size_t size, size_t nitems, void *arg);
int tool_mime_stdin_seek(void *instream, curl_off_t offset, int whence);

int formparse(struct OperationConfig *config,
              const char *input,
              struct tool_mime **mimeroot,
              struct tool_mime **mimecurrent,
              bool literal_value);
CURLcode tool2curlmime(CURL *curl, struct tool_mime *m, curl_mime **mime);
void tool_mime_free(struct tool_mime *mime);

#endif /* HEADER_CURL_TOOL_FORMPARSE_H */
