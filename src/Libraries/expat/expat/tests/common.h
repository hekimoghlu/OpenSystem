/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 25, 2023.
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
#ifdef __cplusplus
extern "C" {
#endif

#ifndef XML_COMMON_H
#  define XML_COMMON_H

#  include "expat_config.h"
#  include "minicheck.h"
#  include "chardata.h"

#  ifdef XML_LARGE_SIZE
#    define XML_FMT_INT_MOD "ll"
#  else
#    define XML_FMT_INT_MOD "l"
#  endif

#  ifdef XML_UNICODE_WCHAR_T
#    define XML_FMT_STR "ls"
#    include <wchar.h>
#    define xcstrlen(s) wcslen(s)
#    define xcstrcmp(s, t) wcscmp((s), (t))
#    define xcstrncmp(s, t, n) wcsncmp((s), (t), (n))
#    define XCS(s) _XCS(s)
#    define _XCS(s) L##s
#  else
#    ifdef XML_UNICODE
#      error "No support for UTF-16 character without wchar_t in tests"
#    else
#      define XML_FMT_STR "s"
#      define xcstrlen(s) strlen(s)
#      define xcstrcmp(s, t) strcmp((s), (t))
#      define xcstrncmp(s, t, n) strncmp((s), (t), (n))
#      define XCS(s) s
#    endif /* XML_UNICODE */
#  endif   /* XML_UNICODE_WCHAR_T */

extern XML_Parser g_parser;

extern XML_Bool g_resumable;
extern XML_Bool g_abortable;

extern int g_chunkSize;

extern const char *long_character_data_text;
extern const char *long_cdata_text;
extern const char *get_buffer_test_text;

extern void tcase_add_test__ifdef_xml_dtd(TCase *tc, tcase_test_function test);
extern void tcase_add_test__if_xml_ge(TCase *tc, tcase_test_function test);

extern void basic_teardown(void);

extern void _xml_failure(XML_Parser parser, const char *file, int line);

#  define xml_failure(parser) _xml_failure((parser), __FILE__, __LINE__)

extern enum XML_Status _XML_Parse_SINGLE_BYTES(XML_Parser parser, const char *s,
                                               int len, int isFinal);

extern void _expect_failure(const char *text, enum XML_Error errorCode,
                            const char *errorMessage, const char *file,
                            int lineno);

#  define expect_failure(text, errorCode, errorMessage)                        \
    _expect_failure((text), (errorCode), (errorMessage), __FILE__, __LINE__)

/* Support functions for handlers to collect up character and attribute data.
 */

extern void _run_character_check(const char *text, const XML_Char *expected,
                                 const char *file, int line);

#  define run_character_check(text, expected)                                  \
    _run_character_check(text, expected, __FILE__, __LINE__)

extern void _run_attribute_check(const char *text, const XML_Char *expected,
                                 const char *file, int line);

#  define run_attribute_check(text, expected)                                  \
    _run_attribute_check(text, expected, __FILE__, __LINE__)

typedef struct ExtTest {
  const char *parse_text;
  const XML_Char *encoding;
  CharData *storage;
} ExtTest;

extern void _run_ext_character_check(const char *text, ExtTest *test_data,
                                     const XML_Char *expected, const char *file,
                                     int line);

#  define run_ext_character_check(text, test_data, expected)                   \
    _run_ext_character_check(text, test_data, expected, __FILE__, __LINE__)

#  define ALLOC_ALWAYS_SUCCEED (-1)
#  define REALLOC_ALWAYS_SUCCEED (-1)

extern int g_allocation_count;
extern int g_reallocation_count;

extern void *duff_allocator(size_t size);

extern void *duff_reallocator(void *ptr, size_t size);

extern char *portable_strndup(const char *s, size_t n);

#endif /* XML_COMMON_H */

#ifdef __cplusplus
}
#endif
