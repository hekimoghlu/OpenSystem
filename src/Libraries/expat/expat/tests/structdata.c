/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 20, 2024.
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

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "structdata.h"
#include "minicheck.h"

#define STRUCT_EXTENSION_COUNT 8

#ifdef XML_UNICODE_WCHAR_T
#  include <wchar.h>
#  define XML_FMT_STR "ls"
#  define xcstrlen(s) wcslen(s)
#  define xcstrcmp(s, t) wcscmp((s), (t))
#else
#  define XML_FMT_STR "s"
#  define xcstrlen(s) strlen(s)
#  define xcstrcmp(s, t) strcmp((s), (t))
#endif

static XML_Char *
xmlstrdup(const XML_Char *s) {
  size_t byte_count = (xcstrlen(s) + 1) * sizeof(XML_Char);
  XML_Char *const dup = (XML_Char *)malloc(byte_count);

  assert(dup != NULL);
  memcpy(dup, s, byte_count);
  return dup;
}

void
StructData_Init(StructData *storage) {
  assert(storage != NULL);
  storage->count = 0;
  storage->max_count = 0;
  storage->entries = NULL;
}

void
StructData_AddItem(StructData *storage, const XML_Char *s, int data0, int data1,
                   int data2) {
  StructDataEntry *entry;

  assert(storage != NULL);
  assert(s != NULL);
  if (storage->count == storage->max_count) {
    StructDataEntry *new_entries;

    storage->max_count += STRUCT_EXTENSION_COUNT;
    new_entries = (StructDataEntry *)realloc(
        storage->entries, storage->max_count * sizeof(StructDataEntry));
    assert(new_entries != NULL);
    storage->entries = new_entries;
  }

  entry = &storage->entries[storage->count];
  entry->str = xmlstrdup(s);
  entry->data0 = data0;
  entry->data1 = data1;
  entry->data2 = data2;
  storage->count++;
}

/* 'fail()' aborts the function via a longjmp, so there is no point
 * in returning a value from this function.
 */
void
StructData_CheckItems(StructData *storage, const StructDataEntry *expected,
                      int count) {
  char buffer[1024];

  assert(storage != NULL);
  assert(expected != NULL);
  if (count != storage->count) {
    snprintf(buffer, sizeof(buffer),
             "wrong number of entries: got %d, expected %d", storage->count,
             count);
    StructData_Dispose(storage);
    fail(buffer);
  } else {
    for (int i = 0; i < count; i++) {
      const StructDataEntry *got = &storage->entries[i];
      const StructDataEntry *want = &expected[i];

      assert(got != NULL);
      assert(want != NULL);

      if (xcstrcmp(got->str, want->str) != 0) {
        StructData_Dispose(storage);
        fail("structure got bad string");
      } else {
        if (got->data0 != want->data0 || got->data1 != want->data1
            || got->data2 != want->data2) {
          snprintf(buffer, sizeof(buffer),
                   "struct '%" XML_FMT_STR
                   "' expected (%d,%d,%d), got (%d,%d,%d)",
                   got->str, want->data0, want->data1, want->data2, got->data0,
                   got->data1, got->data2);
          StructData_Dispose(storage);
          fail(buffer);
        }
      }
    }
  }
}

void
StructData_Dispose(StructData *storage) {
  int i;

  assert(storage != NULL);
  for (i = 0; i < storage->count; i++)
    free((void *)storage->entries[i].str);
  free(storage->entries);

  storage->count = 0;
  storage->entries = NULL;
}

