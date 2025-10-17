/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 7, 2023.
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
#if defined(NDEBUG)
#  undef NDEBUG /* because test suite relies on assert(...) at the moment */
#endif

#include "expat_config.h"
#include "minicheck.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "chardata.h"

static int
xmlstrlen(const XML_Char *s) {
  int len = 0;
  assert(s != NULL);
  while (s[len] != 0)
    ++len;
  return len;
}

void
CharData_Init(CharData *storage) {
  assert(storage != NULL);
  storage->count = -1;
}

void
CharData_AppendXMLChars(CharData *storage, const XML_Char *s, int len) {
  int maxchars;

  assert(storage != NULL);
  assert(s != NULL);
  maxchars = sizeof(storage->data) / sizeof(storage->data[0]);
  if (storage->count < 0)
    storage->count = 0;
  if (len < 0)
    len = xmlstrlen(s);
  if ((len + storage->count) > maxchars) {
    len = (maxchars - storage->count);
  }
  if (len + storage->count < (int)sizeof(storage->data)) {
    memcpy(storage->data + storage->count, s, len * sizeof(storage->data[0]));
    storage->count += len;
  }
}

int
CharData_CheckXMLChars(CharData *storage, const XML_Char *expected) {
  int len = xmlstrlen(expected);
  int count;

  assert(storage != NULL);
  count = (storage->count < 0) ? 0 : storage->count;
  if (len != count) {
    char buffer[1024];
    snprintf(buffer, sizeof(buffer),
             "wrong number of data characters: got %d, expected %d", count,
             len);
    fail(buffer);
    return 0;
  }
  if (memcmp(expected, storage->data, len * sizeof(storage->data[0])) != 0) {
    fail("got bad data bytes");
    return 0;
  }
  return 1;
}

