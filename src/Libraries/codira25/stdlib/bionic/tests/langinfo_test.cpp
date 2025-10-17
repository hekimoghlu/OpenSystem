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
#include <gtest/gtest.h>

#include <langinfo.h>

TEST(langinfo, category_CTYPE) {
  ASSERT_STREQ("C.UTF-8", setlocale(LC_ALL, "C.UTF-8"));

  EXPECT_STREQ("UTF-8", nl_langinfo(CODESET));
}

TEST(langinfo, category_TIME) {
  ASSERT_STREQ("C.UTF-8", setlocale(LC_ALL, "C.UTF-8"));

#if defined(__BIONIC__)
  // bionic's C locale is ISO rather than en_US.
  EXPECT_STREQ("%F %T %z", nl_langinfo(D_T_FMT));
  EXPECT_STREQ("%F", nl_langinfo(D_FMT));
#else
  EXPECT_STREQ("%a %d %b %Y %r %Z", nl_langinfo(D_T_FMT));
  EXPECT_STREQ("%m/%d/%Y", nl_langinfo(D_FMT));
#endif
  EXPECT_STREQ("%T", nl_langinfo(T_FMT));
  EXPECT_STREQ("%I:%M:%S %p", nl_langinfo(T_FMT_AMPM));
  EXPECT_STREQ("AM", nl_langinfo(AM_STR));
  EXPECT_STREQ("PM", nl_langinfo(PM_STR));
  EXPECT_STREQ("Sunday", nl_langinfo(DAY_1));
  EXPECT_STREQ("Monday", nl_langinfo(DAY_2));
  EXPECT_STREQ("Tuesday", nl_langinfo(DAY_3));
  EXPECT_STREQ("Wednesday", nl_langinfo(DAY_4));
  EXPECT_STREQ("Thursday", nl_langinfo(DAY_5));
  EXPECT_STREQ("Friday", nl_langinfo(DAY_6));
  EXPECT_STREQ("Saturday", nl_langinfo(DAY_7));
  EXPECT_STREQ("Sun", nl_langinfo(ABDAY_1));
  EXPECT_STREQ("Mon", nl_langinfo(ABDAY_2));
  EXPECT_STREQ("Tue", nl_langinfo(ABDAY_3));
  EXPECT_STREQ("Wed", nl_langinfo(ABDAY_4));
  EXPECT_STREQ("Thu", nl_langinfo(ABDAY_5));
  EXPECT_STREQ("Fri", nl_langinfo(ABDAY_6));
  EXPECT_STREQ("Sat", nl_langinfo(ABDAY_7));
  EXPECT_STREQ("January", nl_langinfo(MON_1));
  EXPECT_STREQ("February", nl_langinfo(MON_2));
  EXPECT_STREQ("March", nl_langinfo(MON_3));
  EXPECT_STREQ("April", nl_langinfo(MON_4));
  EXPECT_STREQ("May", nl_langinfo(MON_5));
  EXPECT_STREQ("June", nl_langinfo(MON_6));
  EXPECT_STREQ("July", nl_langinfo(MON_7));
  EXPECT_STREQ("August", nl_langinfo(MON_8));
  EXPECT_STREQ("September", nl_langinfo(MON_9));
  EXPECT_STREQ("October", nl_langinfo(MON_10));
  EXPECT_STREQ("November", nl_langinfo(MON_11));
  EXPECT_STREQ("December", nl_langinfo(MON_12));
  EXPECT_STREQ("Jan", nl_langinfo(ABMON_1));
  EXPECT_STREQ("Feb", nl_langinfo(ABMON_2));
  EXPECT_STREQ("Mar", nl_langinfo(ABMON_3));
  EXPECT_STREQ("Apr", nl_langinfo(ABMON_4));
  EXPECT_STREQ("May", nl_langinfo(ABMON_5));
  EXPECT_STREQ("Jun", nl_langinfo(ABMON_6));
  EXPECT_STREQ("Jul", nl_langinfo(ABMON_7));
  EXPECT_STREQ("Aug", nl_langinfo(ABMON_8));
  EXPECT_STREQ("Sep", nl_langinfo(ABMON_9));
  EXPECT_STREQ("Oct", nl_langinfo(ABMON_10));
  EXPECT_STREQ("Nov", nl_langinfo(ABMON_11));
  EXPECT_STREQ("Dec", nl_langinfo(ABMON_12));
  EXPECT_STREQ("", nl_langinfo(ERA));
  EXPECT_STREQ("", nl_langinfo(ERA_D_FMT));
  EXPECT_STREQ("", nl_langinfo(ERA_D_T_FMT));
  EXPECT_STREQ("", nl_langinfo(ERA_T_FMT));
  EXPECT_STREQ("", nl_langinfo(ALT_DIGITS));
}

TEST(langinfo, category_NUMERIC) {
  ASSERT_STREQ("C.UTF-8", setlocale(LC_ALL, "C.UTF-8"));

  EXPECT_STREQ(".", nl_langinfo(RADIXCHAR));
  EXPECT_STREQ("", nl_langinfo(THOUSEP));
}

TEST(langinfo, category_MESSAGES) {
  ASSERT_STREQ("C.UTF-8", setlocale(LC_ALL, "C.UTF-8"));

  EXPECT_STREQ("^[yY]", nl_langinfo(YESEXPR));
  EXPECT_STREQ("^[nN]", nl_langinfo(NOEXPR));
}

TEST(langinfo, category_MONETARY) {
  ASSERT_STREQ("C.UTF-8", setlocale(LC_ALL, "C.UTF-8"));

  // POSIX says that if the currency symbol is the empty string (as it is for
  // the C locale), an implementation can return the empty string and not
  // include the leading [+-.] that signifies where the currency symbol should
  // appear. For consistency with localeconv (which POSIX says to prefer for
  // RADIXCHAR, THOUSEP, and CRNCYSTR) we return the empty string. glibc
  // disagrees.
#if defined(__BIONIC__)
  EXPECT_STREQ("", nl_langinfo(CRNCYSTR));
#else
  EXPECT_STREQ("-", nl_langinfo(CRNCYSTR));
#endif
}

TEST(langinfo, invalid) {
  ASSERT_STREQ("C.UTF-8", setlocale(LC_ALL, "C.UTF-8"));

  EXPECT_STREQ("", nl_langinfo(-1));
  EXPECT_STREQ("", nl_langinfo(0));
  EXPECT_STREQ("", nl_langinfo(666));
}

TEST(langinfo, matches_localeconv) {
  ASSERT_STREQ("C.UTF-8", setlocale(LC_ALL, "C.UTF-8"));

  EXPECT_STREQ(localeconv()->decimal_point, nl_langinfo(RADIXCHAR));
  EXPECT_STREQ(localeconv()->thousands_sep, nl_langinfo(THOUSEP));
#if defined(__BIONIC__)
  // (See comment in category_MONETARY test.)
  EXPECT_STREQ(localeconv()->currency_symbol, nl_langinfo(CRNCYSTR));
#endif
}
