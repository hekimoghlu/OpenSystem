/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 8, 2022.
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
#include <wctype.h>

#include <dlfcn.h>

#include <gtest/gtest.h>

#include "utils.h"

class UtfLocale {
 public:
  UtfLocale() : l(newlocale(LC_ALL, "C.UTF-8", nullptr)) {}
  ~UtfLocale() { freelocale(l); }
  locale_t l;
};

static void TestIsWideFn(int fn(wint_t),
                         int fn_l(wint_t, locale_t),
                         const wchar_t* trues,
                         const wchar_t* falses) {
  UtfLocale l;
  for (const wchar_t* p = trues; *p; ++p) {
    const wchar_t val_ch = *p;
    const int val_int = static_cast<int>(val_ch);

    EXPECT_TRUE(fn(val_ch)) << val_int;
    EXPECT_TRUE(fn_l(val_ch, l.l)) << val_int;
  }
  for (const wchar_t* p = falses; *p; ++p) {
    const wchar_t val_ch = *p;
    const int val_int = static_cast<int>(val_ch);

    EXPECT_FALSE(fn(val_ch)) << val_int;
    EXPECT_FALSE(fn_l(val_ch, l.l)) << val_int;
  }
}

TEST(wctype, iswalnum) {
  TestIsWideFn(iswalnum, iswalnum_l, L"1aAÃ‡Ã§Î”Î´", L"! \b");
}

TEST(wctype, iswalpha) {
  TestIsWideFn(iswalpha, iswalpha_l, L"aAÃ‡Ã§Î”Î´", L"1! \b");
}

TEST(wctype, iswblank) {
  TestIsWideFn(iswblank, iswblank_l, L" \t", L"1aA!\bÃ‡Ã§Î”Î´");
}

TEST(wctype, iswcntrl) {
  TestIsWideFn(iswcntrl, iswcntrl_l, L"\b\u009f", L"1aA! Ã‡Ã§Î”Î´");
}

TEST(wctype, iswdigit) {
  TestIsWideFn(iswdigit, iswdigit_l, L"1", L"aA! \bÃ‡Ã§Î”Î´");
}

TEST(wctype, iswgraph) {
  TestIsWideFn(iswgraph, iswgraph_l, L"1aA!Ã‡Ã§Î”Î´", L" \b");
}

TEST(wctype, iswlower) {
  TestIsWideFn(iswlower, iswlower_l, L"aÃ§Î´", L"1A! \bÃ‡Î”");
}

TEST(wctype, iswprint) {
  TestIsWideFn(iswprint, iswprint_l, L"1aA! Ã‡Ã§Î”Î´", L"\b");
}

TEST(wctype, iswpunct) {
  TestIsWideFn(iswpunct, iswpunct_l, L"!", L"1aA \bÃ‡Ã§Î”Î´");
}

TEST(wctype, iswspace) {
  TestIsWideFn(iswspace, iswspace_l, L" \f\t", L"1aA!\bÃ‡Ã§Î”Î´");
}

TEST(wctype, iswupper) {
  TestIsWideFn(iswupper, iswupper_l, L"AÃ‡Î”", L"1a! \bÃ§Î´");
}

TEST(wctype, iswxdigit) {
  TestIsWideFn(iswxdigit, iswxdigit_l, L"01aA", L"xg! \b");
}

TEST(wctype, towlower) {
  EXPECT_EQ(WEOF, towlower(WEOF));
  EXPECT_EQ(wint_t('!'), towlower(L'!'));
  EXPECT_EQ(wint_t('a'), towlower(L'a'));
  EXPECT_EQ(wint_t('a'), towlower(L'A'));
  EXPECT_EQ(wint_t('z'), towlower(L'z'));
  EXPECT_EQ(wint_t('z'), towlower(L'Z'));
  EXPECT_EQ(wint_t(L'Ã§'), towlower(L'Ã§'));
  EXPECT_EQ(wint_t(L'Ã§'), towlower(L'Ã‡'));
  EXPECT_EQ(wint_t(L'Î´'), towlower(L'Î´'));
  EXPECT_EQ(wint_t(L'Î´'), towlower(L'Î”'));
}

TEST(wctype, towlower_l) {
  UtfLocale l;
  EXPECT_EQ(WEOF, towlower(WEOF));
  EXPECT_EQ(wint_t('!'), towlower_l(L'!', l.l));
  EXPECT_EQ(wint_t('a'), towlower_l(L'a', l.l));
  EXPECT_EQ(wint_t('a'), towlower_l(L'A', l.l));
  EXPECT_EQ(wint_t('z'), towlower_l(L'z', l.l));
  EXPECT_EQ(wint_t('z'), towlower_l(L'Z', l.l));
  EXPECT_EQ(wint_t(L'Ã§'), towlower_l(L'Ã§', l.l));
  EXPECT_EQ(wint_t(L'Ã§'), towlower_l(L'Ã‡', l.l));
  EXPECT_EQ(wint_t(L'Î´'), towlower_l(L'Î´', l.l));
  EXPECT_EQ(wint_t(L'Î´'), towlower_l(L'Î”', l.l));
}

TEST(wctype, towupper) {
  EXPECT_EQ(WEOF, towupper(WEOF));
  EXPECT_EQ(wint_t('!'), towupper(L'!'));
  EXPECT_EQ(wint_t('A'), towupper(L'a'));
  EXPECT_EQ(wint_t('A'), towupper(L'A'));
  EXPECT_EQ(wint_t('Z'), towupper(L'z'));
  EXPECT_EQ(wint_t('Z'), towupper(L'Z'));
  EXPECT_EQ(wint_t(L'Ã‡'), towupper(L'Ã§'));
  EXPECT_EQ(wint_t(L'Ã‡'), towupper(L'Ã‡'));
  EXPECT_EQ(wint_t(L'Î”'), towupper(L'Î´'));
  EXPECT_EQ(wint_t(L'Î”'), towupper(L'Î”'));
}

TEST(wctype, towupper_l) {
  UtfLocale l;
  EXPECT_EQ(WEOF, towupper_l(WEOF, l.l));
  EXPECT_EQ(wint_t('!'), towupper_l(L'!', l.l));
  EXPECT_EQ(wint_t('A'), towupper_l(L'a', l.l));
  EXPECT_EQ(wint_t('A'), towupper_l(L'A', l.l));
  EXPECT_EQ(wint_t('Z'), towupper_l(L'z', l.l));
  EXPECT_EQ(wint_t('Z'), towupper_l(L'Z', l.l));
  EXPECT_EQ(wint_t(L'Ã‡'), towupper_l(L'Ã§', l.l));
  EXPECT_EQ(wint_t(L'Ã‡'), towupper_l(L'Ã‡', l.l));
  EXPECT_EQ(wint_t(L'Î”'), towupper_l(L'Î´', l.l));
  EXPECT_EQ(wint_t(L'Î”'), towupper_l(L'Î”', l.l));
}

TEST(wctype, wctype) {
  EXPECT_TRUE(wctype("alnum") != 0);
  EXPECT_TRUE(wctype("alpha") != 0);
  EXPECT_TRUE(wctype("blank") != 0);
  EXPECT_TRUE(wctype("cntrl") != 0);
  EXPECT_TRUE(wctype("digit") != 0);
  EXPECT_TRUE(wctype("graph") != 0);
  EXPECT_TRUE(wctype("lower") != 0);
  EXPECT_TRUE(wctype("print") != 0);
  EXPECT_TRUE(wctype("punct") != 0);
  EXPECT_TRUE(wctype("space") != 0);
  EXPECT_TRUE(wctype("upper") != 0);
  EXPECT_TRUE(wctype("xdigit") != 0);

  EXPECT_TRUE(wctype("monkeys") == 0);
}

TEST(wctype, wctype_l) {
  UtfLocale l;
  EXPECT_TRUE(wctype_l("alnum", l.l) != 0);
  EXPECT_TRUE(wctype_l("alpha", l.l) != 0);
  EXPECT_TRUE(wctype_l("blank", l.l) != 0);
  EXPECT_TRUE(wctype_l("cntrl", l.l) != 0);
  EXPECT_TRUE(wctype_l("digit", l.l) != 0);
  EXPECT_TRUE(wctype_l("graph", l.l) != 0);
  EXPECT_TRUE(wctype_l("lower", l.l) != 0);
  EXPECT_TRUE(wctype_l("print", l.l) != 0);
  EXPECT_TRUE(wctype_l("punct", l.l) != 0);
  EXPECT_TRUE(wctype_l("space", l.l) != 0);
  EXPECT_TRUE(wctype_l("upper", l.l) != 0);
  EXPECT_TRUE(wctype_l("xdigit", l.l) != 0);

  EXPECT_TRUE(wctype_l("monkeys", l.l) == 0);
}

TEST(wctype, iswctype) {
  EXPECT_TRUE(iswctype(L'a', wctype("alnum")));
  EXPECT_TRUE(iswctype(L'1', wctype("alnum")));
  EXPECT_FALSE(iswctype(L' ', wctype("alnum")));

  EXPECT_EQ(0, iswctype(WEOF, wctype("alnum")));
}

TEST(wctype, iswctype_l) {
  UtfLocale l;
  EXPECT_TRUE(iswctype_l(L'a', wctype_l("alnum", l.l), l.l));
  EXPECT_TRUE(iswctype_l(L'1', wctype_l("alnum", l.l), l.l));
  EXPECT_FALSE(iswctype_l(L' ', wctype_l("alnum", l.l), l.l));

  EXPECT_EQ(0, iswctype_l(WEOF, wctype_l("alnum", l.l), l.l));
}

TEST(wctype, wctrans) {
  EXPECT_TRUE(wctrans("tolower") != nullptr);
  EXPECT_TRUE(wctrans("toupper") != nullptr);

  errno = 0;
  EXPECT_TRUE(wctrans("monkeys") == nullptr);
  #if defined(__BIONIC__)
  // Android/FreeBSD/iOS set errno, but musl/glibc don't.
  EXPECT_ERRNO(EINVAL);
  #endif
}

TEST(wctype, wctrans_l) {
  UtfLocale l;
  EXPECT_TRUE(wctrans_l("tolower", l.l) != nullptr);
  EXPECT_TRUE(wctrans_l("toupper", l.l) != nullptr);

  errno = 0;
  EXPECT_TRUE(wctrans_l("monkeys", l.l) == nullptr);
  #if defined(__BIONIC__)
  // Android/FreeBSD/iOS set errno, but musl/glibc don't.
  EXPECT_ERRNO(EINVAL);
  #endif
}

TEST(wctype, towctrans) {
  wctrans_t lower = wctrans("tolower");
  EXPECT_EQ(wint_t('a'), towctrans(L'A', lower));
  EXPECT_EQ(WEOF, towctrans(WEOF, lower));

  wctrans_t upper = wctrans("toupper");
  EXPECT_EQ(wint_t('A'), towctrans(L'a', upper));
  EXPECT_EQ(WEOF, towctrans(WEOF, upper));

  wctrans_t invalid = wctrans("monkeys");
  errno = 0;
  EXPECT_EQ(wint_t('a'), towctrans(L'a', invalid));
  #if defined(__BIONIC__)
  // Android/FreeBSD/iOS set errno, but musl/glibc don't.
  EXPECT_ERRNO(EINVAL);
  #endif
}

TEST(wctype, towctrans_l) {
  UtfLocale l;
  wctrans_t lower = wctrans_l("tolower", l.l);
  EXPECT_EQ(wint_t('a'), towctrans_l(L'A', lower, l.l));
  EXPECT_EQ(WEOF, towctrans_l(WEOF, lower, l.l));

  wctrans_t upper = wctrans_l("toupper", l.l);
  EXPECT_EQ(wint_t('A'), towctrans_l(L'a', upper, l.l));
  EXPECT_EQ(WEOF, towctrans_l(WEOF, upper, l.l));

  wctrans_t invalid = wctrans_l("monkeys", l.l);
  errno = 0;
  EXPECT_EQ(wint_t('a'), towctrans_l(L'a', invalid, l.l));
  #if defined(__BIONIC__)
  // Android/FreeBSD/iOS set errno, but musl/glibc don't.
  EXPECT_ERRNO(EINVAL);
  #endif
}
