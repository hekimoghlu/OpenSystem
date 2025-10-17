/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 26, 2023.
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
#include <errno.h>
#include <locale.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <wchar.h>

#include "bionic/pthread_internal.h"
#include "platform/bionic/macros.h"

// We only support two locales, the "C" locale (also known as "POSIX"),
// and the "C.UTF-8" locale (also known as "en_US.UTF-8").

static bool __bionic_current_locale_is_utf8 = true;

struct __locale_t {
  size_t mb_cur_max;
};

// Avoid using new/delete in this file, because a user may have overridden
// new/delete, and we want to avoid making extraneous calls to them. This isn't
// an issue for libc.so in the platform, but this file is also compiled into the
// NDK's libandroid_support.a, and there are libc++ tests that count the number
// of calls to new/delete.
#pragma clang poison new delete

static inline locale_t __alloc_locale(size_t mb_cur_max) {
  auto result = static_cast<__locale_t*>(malloc(sizeof(__locale_t)));
  if (result == nullptr) return nullptr;
  result->mb_cur_max = mb_cur_max;
  return result;
}

static inline size_t get_locale_mb_cur_max(locale_t l) {
  if (l == LC_GLOBAL_LOCALE) {
    return __bionic_current_locale_is_utf8 ? 4 : 1;
  } else {
    return l->mb_cur_max;
  }
}

size_t __ctype_get_mb_cur_max() {
  return get_locale_mb_cur_max(uselocale(nullptr));
}

static pthread_once_t g_locale_once = PTHREAD_ONCE_INIT;
static lconv g_locale;

static void __locale_init() {
  g_locale.decimal_point = const_cast<char*>(".");

  char* not_available = const_cast<char*>("");
  g_locale.thousands_sep = not_available;
  g_locale.grouping = not_available;
  g_locale.int_curr_symbol = not_available;
  g_locale.currency_symbol = not_available;
  g_locale.mon_decimal_point = not_available;
  g_locale.mon_thousands_sep = not_available;
  g_locale.mon_grouping = not_available;
  g_locale.positive_sign = not_available;
  g_locale.negative_sign = not_available;

  g_locale.int_frac_digits = CHAR_MAX;
  g_locale.frac_digits = CHAR_MAX;
  g_locale.p_cs_precedes = CHAR_MAX;
  g_locale.p_sep_by_space = CHAR_MAX;
  g_locale.n_cs_precedes = CHAR_MAX;
  g_locale.n_sep_by_space = CHAR_MAX;
  g_locale.p_sign_posn = CHAR_MAX;
  g_locale.n_sign_posn = CHAR_MAX;
  g_locale.int_p_cs_precedes = CHAR_MAX;
  g_locale.int_p_sep_by_space = CHAR_MAX;
  g_locale.int_n_cs_precedes = CHAR_MAX;
  g_locale.int_n_sep_by_space = CHAR_MAX;
  g_locale.int_p_sign_posn = CHAR_MAX;
  g_locale.int_n_sign_posn = CHAR_MAX;
}

static bool __is_supported_locale(const char* locale_name) {
  return (strcmp(locale_name, "") == 0 ||
          strcmp(locale_name, "C") == 0 ||
          strcmp(locale_name, "C.UTF-8") == 0 ||
          strcmp(locale_name, "en_US.UTF-8") == 0 ||
          strcmp(locale_name, "POSIX") == 0);
}

static bool __is_utf8_locale(const char* locale_name) {
  return (*locale_name == '\0' || strstr(locale_name, "UTF-8"));
}

lconv* localeconv() {
  pthread_once(&g_locale_once, __locale_init);
  return &g_locale;
}

locale_t duplocale(locale_t l) {
  return __alloc_locale(get_locale_mb_cur_max(l));
}

void freelocale(locale_t l) {
  free(l);
}

locale_t newlocale(int category_mask, const char* locale_name, locale_t /*base*/) {
  // Are 'category_mask' and 'locale_name' valid?
  if ((category_mask & ~LC_ALL_MASK) != 0 || locale_name == nullptr) {
    errno = EINVAL;
    return nullptr;
  }

  if (!__is_supported_locale(locale_name)) {
    errno = ENOENT;
    return nullptr;
  }

  return __alloc_locale(__is_utf8_locale(locale_name) ? 4 : 1);
}

char* setlocale(int category, const char* locale_name) {
  // Is 'category' valid?
  if (category < LC_CTYPE || category > LC_IDENTIFICATION) {
    errno = EINVAL;
    return nullptr;
  }

  // Caller wants to set the locale rather than just query?
  if (locale_name != nullptr) {
    if (!__is_supported_locale(locale_name)) {
      // We don't support this locale.
      errno = ENOENT;
      return nullptr;
    }
    __bionic_current_locale_is_utf8 = __is_utf8_locale(locale_name);
  }

  return const_cast<char*>(__bionic_current_locale_is_utf8 ? "C.UTF-8" : "C");
}

static locale_t* get_current_locale_ptr() {
  return &__get_bionic_tls().locale;
}

locale_t uselocale(locale_t new_locale) {
  locale_t old_locale = *get_current_locale_ptr();

  // If this is the first call to uselocale(3) on this thread, we return LC_GLOBAL_LOCALE.
  if (old_locale == nullptr) {
    old_locale = LC_GLOBAL_LOCALE;
  }

  if (new_locale != nullptr) {
    *get_current_locale_ptr() = new_locale;
  }

  return old_locale;
}
