/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 3, 2024.
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
#include <assert.h>
#include <errno.h>
#include <stdint.h> // for SIZE_MAX
#include <stdio.h>
#include <string.h>

#include "expat_config.h"
#include "expat.h"
#include "internal.h"
#include "chardata.h"
#include "minicheck.h"
#include "common.h"
#include "handlers.h"

/* Common test data */

const char *long_character_data_text
    = "<?xml version='1.0' encoding='iso-8859-1'?><s>"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "</s>";

const char *long_cdata_text
    = "<s><![CDATA["
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "012345678901234567890123456789012345678901234567890123456789"
      "]]></s>";

/* Having an element name longer than 1024 characters exercises some
 * of the pool allocation code in the parser that otherwise does not
 * get executed.  The count at the end of the line is the number of
 * characters (bytes) in the element name by that point.x
 */
const char *get_buffer_test_text
    = "<documentwitharidiculouslylongelementnametotease"  /* 0x030 */
      "aparticularcorneroftheallocationinXML_GetBuffers"  /* 0x060 */
      "othatwecanimprovethecoverageyetagain012345678901"  /* 0x090 */
      "123456789abcdef0123456789abcdef0123456789abcdef0"  /* 0x0c0 */
      "123456789abcdef0123456789abcdef0123456789abcdef0"  /* 0x0f0 */
      "123456789abcdef0123456789abcdef0123456789abcdef0"  /* 0x120 */
      "123456789abcdef0123456789abcdef0123456789abcdef0"  /* 0x150 */
      "123456789abcdef0123456789abcdef0123456789abcdef0"  /* 0x180 */
      "123456789abcdef0123456789abcdef0123456789abcdef0"  /* 0x1b0 */
      "123456789abcdef0123456789abcdef0123456789abcdef0"  /* 0x1e0 */
      "123456789abcdef0123456789abcdef0123456789abcdef0"  /* 0x210 */
      "123456789abcdef0123456789abcdef0123456789abcdef0"  /* 0x240 */
      "123456789abcdef0123456789abcdef0123456789abcdef0"  /* 0x270 */
      "123456789abcdef0123456789abcdef0123456789abcdef0"  /* 0x2a0 */
      "123456789abcdef0123456789abcdef0123456789abcdef0"  /* 0x2d0 */
      "123456789abcdef0123456789abcdef0123456789abcdef0"  /* 0x300 */
      "123456789abcdef0123456789abcdef0123456789abcdef0"  /* 0x330 */
      "123456789abcdef0123456789abcdef0123456789abcdef0"  /* 0x360 */
      "123456789abcdef0123456789abcdef0123456789abcdef0"  /* 0x390 */
      "123456789abcdef0123456789abcdef0123456789abcdef0"  /* 0x3c0 */
      "123456789abcdef0123456789abcdef0123456789abcdef0"  /* 0x3f0 */
      "123456789abcdef0123456789abcdef0123456789>\n<ef0"; /* 0x420 */

/* Test control globals */

/* Used as the "resumable" parameter to XML_StopParser by some tests */
XML_Bool g_resumable = XML_FALSE;

/* Used to control abort checks in some tests */
XML_Bool g_abortable = XML_FALSE;

/* Used to control _XML_Parse_SINGLE_BYTES() chunk size */
int g_chunkSize = 1;

/* Common test functions */

void
tcase_add_test__ifdef_xml_dtd(TCase *tc, tcase_test_function test) {
#ifdef XML_DTD
  tcase_add_test(tc, test);
#else
  UNUSED_P(tc);
  UNUSED_P(test);
#endif
}

void
tcase_add_test__if_xml_ge(TCase *tc, tcase_test_function test) {
#if XML_GE == 1
  tcase_add_test(tc, test);
#else
  UNUSED_P(tc);
  UNUSED_P(test);
#endif
}

void
basic_teardown(void) {
  if (g_parser != NULL) {
    XML_ParserFree(g_parser);
    g_parser = NULL;
  }
}

/* Generate a failure using the parser state to create an error message;
   this should be used when the parser reports an error we weren't
   expecting.
*/
void
_xml_failure(XML_Parser parser, const char *file, int line) {
  char buffer[1024];
  enum XML_Error err = XML_GetErrorCode(parser);
  snprintf(buffer, sizeof(buffer),
           "    %d: %" XML_FMT_STR " (line %" XML_FMT_INT_MOD
           "u, offset %" XML_FMT_INT_MOD "u)\n    reported from %s, line %d\n",
           err, XML_ErrorString(err), XML_GetCurrentLineNumber(parser),
           XML_GetCurrentColumnNumber(parser), file, line);
  _fail(file, line, buffer);
}

enum XML_Status
_XML_Parse_SINGLE_BYTES(XML_Parser parser, const char *s, int len,
                        int isFinal) {
  // This ensures that tests have to run pathological parse cases
  // (e.g. when `s` is NULL) against plain XML_Parse rather than
  // chunking _XML_Parse_SINGLE_BYTES.
  assert((parser != NULL) && (s != NULL) && (len >= 0));
  const int chunksize = g_chunkSize;
  if (chunksize > 0) {
    // parse in chunks of `chunksize` bytes as long as not exhausting
    for (; len > chunksize; len -= chunksize, s += chunksize) {
      enum XML_Status res = XML_Parse(parser, s, chunksize, XML_FALSE);
      if (res != XML_STATUS_OK) {
        if ((res == XML_STATUS_SUSPENDED) && (len > chunksize)) {
          fail("Use of function _XML_Parse_SINGLE_BYTES with a chunk size "
               "greater than 0 (from g_chunkSize) does not work well with "
               "suspension. Please consider use of plain XML_Parse at this "
               "place in your test, instead.");
        }
        return res;
      }
    }
  }
  // parse the final chunk, the size of which will be <= chunksize
  return XML_Parse(parser, s, len, isFinal);
}

void
_expect_failure(const char *text, enum XML_Error errorCode,
                const char *errorMessage, const char *file, int lineno) {
  if (_XML_Parse_SINGLE_BYTES(g_parser, text, (int)strlen(text), XML_TRUE)
      == XML_STATUS_OK)
    /* Hackish use of _fail() macro, but lets us report
       the right filename and line number. */
    _fail(file, lineno, errorMessage);
  if (XML_GetErrorCode(g_parser) != errorCode)
    _xml_failure(g_parser, file, lineno);
}

void
_run_character_check(const char *text, const XML_Char *expected,
                     const char *file, int line) {
  CharData storage;

  CharData_Init(&storage);
  XML_SetUserData(g_parser, &storage);
  XML_SetCharacterDataHandler(g_parser, accumulate_characters);
  if (_XML_Parse_SINGLE_BYTES(g_parser, text, (int)strlen(text), XML_TRUE)
      == XML_STATUS_ERROR)
    _xml_failure(g_parser, file, line);
  CharData_CheckXMLChars(&storage, expected);
}

void
_run_attribute_check(const char *text, const XML_Char *expected,
                     const char *file, int line) {
  CharData storage;

  CharData_Init(&storage);
  XML_SetUserData(g_parser, &storage);
  XML_SetStartElementHandler(g_parser, accumulate_attribute);
  if (_XML_Parse_SINGLE_BYTES(g_parser, text, (int)strlen(text), XML_TRUE)
      == XML_STATUS_ERROR)
    _xml_failure(g_parser, file, line);
  CharData_CheckXMLChars(&storage, expected);
}

void
_run_ext_character_check(const char *text, ExtTest *test_data,
                         const XML_Char *expected, const char *file, int line) {
  CharData *const storage = (CharData *)malloc(sizeof(CharData));

  CharData_Init(storage);
  test_data->storage = storage;
  XML_SetUserData(g_parser, test_data);
  XML_SetCharacterDataHandler(g_parser, ext_accumulate_characters);
  if (_XML_Parse_SINGLE_BYTES(g_parser, text, (int)strlen(text), XML_TRUE)
      == XML_STATUS_ERROR)
    _xml_failure(g_parser, file, line);
  CharData_CheckXMLChars(storage, expected);

  free(storage);
}

/* Control variable; the number of times duff_allocator() will successfully
 * allocate */
#define ALLOC_ALWAYS_SUCCEED (-1)
#define REALLOC_ALWAYS_SUCCEED (-1)

int g_allocation_count = ALLOC_ALWAYS_SUCCEED;
int g_reallocation_count = REALLOC_ALWAYS_SUCCEED;

/* Crocked allocator for allocation failure tests */
void *
duff_allocator(size_t size) {
  if (g_allocation_count == 0)
    return NULL;
  if (g_allocation_count != ALLOC_ALWAYS_SUCCEED)
    g_allocation_count--;
  return malloc(size);
}

/* Crocked reallocator for allocation failure tests */
void *
duff_reallocator(void *ptr, size_t size) {
  if (g_reallocation_count == 0)
    return NULL;
  if (g_reallocation_count != REALLOC_ALWAYS_SUCCEED)
    g_reallocation_count--;
  return realloc(ptr, size);
}

// Portable remake of strndup(3) for C99; does not care about space efficiency
char *
portable_strndup(const char *s, size_t n) {
  if ((s == NULL) || (n == SIZE_MAX)) {
    errno = EINVAL;
    return NULL;
  }

  char *const buffer = (char *)malloc(n + 1);
  if (buffer == NULL) {
    errno = ENOMEM;
    return NULL;
  }

  errno = 0;

  memcpy(buffer, s, n);

  buffer[n] = '\0';

  return buffer;
}

