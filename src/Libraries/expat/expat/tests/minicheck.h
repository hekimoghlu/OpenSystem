/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 22, 2025.
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

#ifndef XML_MINICHECK_H
#  define XML_MINICHECK_H

#  define CK_NOFORK 0
#  define CK_FORK 1

#  define CK_SILENT 0
#  define CK_NORMAL 1
#  define CK_VERBOSE 2

/* Workaround for Microsoft's compiler and Tru64 Unix systems where the
   C compiler has a working __func__, but the C++ compiler only has a
   working __FUNCTION__.  This could be fixed in configure.in, but it's
   not worth it right now. */
#  if defined(_MSC_VER) || (defined(__osf__) && defined(__cplusplus))
#    define __func__ __FUNCTION__
#  endif

/* PRINTF_LIKE has two effects:
    1. Make clang's -Wformat-nonliteral stop warning about non-literal format
       strings in annotated functions' code.
    2. Make both clang and gcc's -Wformat-nonliteral warn about *callers* of
       the annotated function that use a non-literal format string.
*/
#  if defined(__GNUC__)
#    define PRINTF_LIKE(fmtpos, argspos)                                       \
      __attribute__((format(printf, fmtpos, argspos)))
#  else
#    define PRINTF_LIKE(fmtpos, argspos)
#  endif

#  define START_TEST(testname)                                                 \
    static void testname(void) {                                               \
      _check_set_test_info(__func__, __FILE__, __LINE__);                      \
      {
#  define END_TEST                                                             \
    }                                                                          \
    }

void PRINTF_LIKE(1, 2) set_subtest(char const *fmt, ...);

#  define fail(msg) _fail(__FILE__, __LINE__, msg)
#  define assert_true(cond)                                                    \
    do {                                                                       \
      if (! (cond)) {                                                          \
        _fail(__FILE__, __LINE__, "check failed: " #cond);                     \
      }                                                                        \
    } while (0)

typedef void (*tcase_setup_function)(void);
typedef void (*tcase_teardown_function)(void);
typedef void (*tcase_test_function)(void);

typedef struct SRunner SRunner;
typedef struct Suite Suite;
typedef struct TCase TCase;

struct SRunner {
  Suite *suite;
  int nchecks;
  int nfailures;
};

struct Suite {
  const char *name;
  TCase *tests;
};

struct TCase {
  const char *name;
  tcase_setup_function setup;
  tcase_teardown_function teardown;
  tcase_test_function *tests;
  int ntests;
  int allocated;
  TCase *next_tcase;
};

/* Internal helper. */
void _check_set_test_info(char const *function, char const *filename,
                          int lineno);

/*
 * Prototypes for the actual implementation.
 */

#  if defined(__has_attribute)
#    if __has_attribute(noreturn)
__attribute__((noreturn))
#    endif
#  endif
void
_fail(const char *file, int line, const char *msg);
Suite *suite_create(const char *name);
TCase *tcase_create(const char *name);
void suite_add_tcase(Suite *suite, TCase *tc);
void tcase_add_checked_fixture(TCase *tc, tcase_setup_function setup,
                               tcase_teardown_function teardown);
void tcase_add_test(TCase *tc, tcase_test_function test);
SRunner *srunner_create(Suite *suite);
void srunner_run_all(SRunner *runner, const char *context, int verbosity);
void srunner_summarize(SRunner *runner, int verbosity);
int srunner_ntests_failed(SRunner *runner);
void srunner_free(SRunner *runner);

#endif /* XML_MINICHECK_H */

#ifdef __cplusplus
}
#endif
