/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 2, 2025.
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
#include <ctype.h>
#include <dirent.h>
#include <locale.h>
#include <paths.h>
#include <stdbool.h>
#include <wchar.h>
#include <xlocale.h>

#include <darwintest.h>
#include <TargetConditionals.h>

/* Requiring root for this one because we want leak checking. */
T_DECL(ctype_loadall, "Check loading of all installed locales' LC_CTYPE",
		T_META_ASROOT(true))
{
	DIR *d;
	struct dirent *de;
	const char *locale;

	d = opendir(_PATH_LOCALE);
	T_ASSERT_NOTNULL(d, NULL);

	locale = "C";
	T_QUIET;
	T_ASSERT_EQ_STR(locale, setlocale(LC_CTYPE, locale), NULL);

	locale = "POSIX";
	T_QUIET;
	T_ASSERT_EQ_STR(locale, setlocale(LC_CTYPE, locale), NULL);

	/*
	 * Load every locale in /usr/share/locale for LC_CTYPE.  This test
	 * actually serves two useful functions:
	 *
	 * 1.) Checks that we can still load ctypes for everything we install
	 * 2.) Checks that we properly free memory as we rotate through locales
	 *
	 * It takes at least two locales away from the C locale to know if we
	 * leaked anything or not, since the C locale as located in .data and
	 * the first one will be retained in the cache until the second one
	 * replaces it.
	 */
	for (de = readdir(d); de != NULL; de = readdir(d)) {
		if (strcmp(de->d_name, ".") == 0 ||
		    strcmp(de->d_name, "..") == 0)
			continue;

		locale = de->d_name;

		T_QUIET;
		T_ASSERT_EQ_STR(locale, setlocale(LC_CTYPE, locale), NULL);
	}
}

/*
 * We don't install the necessary localedata on embedded platforms to be able to
 * usefully run these tests, so just limit them to macOS.
 */
#if TARGET_OS_OSX
T_DECL(ctype_runetype, "Check LC_CTYPE:runetype table functionality")
{
	wchar_t c;
	bool chk, legacy;

	T_ASSERT_EQ_STR("en_US.UTF-8", setlocale(LC_CTYPE, "en_US.UTF-8"), NULL);

	/*
	 * First, we'll test one of the simple tables, just a contiguous set of
	 * characters.
	 */
	for (c = 0xf0000 - 1; c <= 0xfffff; c++) {
		chk = iswgraph(c);

		T_QUIET;
		T_EXPECT_EQ(chk, (c >= 0xf0000 && c <= 0xffffd), NULL);
	}

	/* <ROCKET> is punctuation with the old definitions. */
	legacy = iswpunct(0x1f680);

	/*
	 * The range from U+1F680 - U+1F6FF is a little more complicated in the
	 * legacy definitions;  a bulk of these are PRINT/GRAPH/PUNCT, but
	 * there's a subset that are marked SWIDTH1/2 so they should be
	 * represented in type tables separately from the min/map/max
	 * definition.
	 */
	for (c = 0x1f680; c <= 0x1f6ff; c++) {
		if (legacy) {
			bool pgp_expected;
			int width;

			pgp_expected = (c >= 0x1f680 && c <= 0x1f6d2) ||
			    (c >= 0x1f6e0 && c <= 0x1f6ec) ||
			    (c >= 0x1f6f0 && c <= 0x1f6f6);

			T_QUIET;
			T_EXPECT_EQ(iswprint(c), pgp_expected,
			    "0x%x is print", c);

			T_QUIET;
			T_EXPECT_EQ(iswgraph(c), pgp_expected,
			    "0x%x is graph", c);

			T_QUIET;
			T_EXPECT_EQ(iswpunct(c), pgp_expected,
			    "0x%x is punctuation", c);

			width = 1;
			if ((c >= 0x1f680 && c <= 0x1f6c5) || c == 0x1f6cc ||
			    (c >= 0x1f6d0 && c <= 0x1f6d2) || c == 0x1f6eb ||
			    c == 0x1f6ec || (c >= 0x1f6f4 && c <= 0x1f6f6))
				width = 2;
			else if (!iswprint(c))
				width = -1;

			T_QUIET;
			T_EXPECT_EQ(wcwidth(c), width, NULL);
		} else if (!((c >= 0x1f6d8 && c <= 0x1f6df) ||
		    (c >= 0x1f6ed && c <= 0x1f6ef) ||
		    (c >= 0x1f6fd && c <= 0x1f6ff))) {
			int width = 2;

			/*
			 * None of the ranges in the conditional just above this
			 * are currently allocated, so don't assert anything on
			 * them for the time being.
			 */
			T_QUIET;
			T_EXPECT_EQ(iswprint(c), 1, "0x%x is print", c);

			T_QUIET;
			T_EXPECT_EQ(iswgraph(c), 1, "0x%x is graph", c);

			if ((c >= 0x1f6c6 && c <= 0x1f6cf && c != 0x1f6cc) ||
			    (c >= 0x1f6e0 && c <= 0x1f6ea) ||
			    (c >= 0x1f6f0 && c <= 0x1f6f3) ||
			    c == 0x1f6d3 || c == 0x1f6d4)
				width = 1;

			T_QUIET;
			T_EXPECT_EQ(wcwidth(c), width, "0x%x is %d wide", c,
			    width);
		}
	}
}

T_DECL(ctype_maplower, "Check LC_CTYPE:maplower table functionality")
{
	wchar_t c, lc;

	T_ASSERT_EQ_STR("en_US.UTF-8", setlocale(LC_CTYPE, "en_US.UTF-8"), NULL);

	/*
	 * For maplower and mapupper, we'll only check a limited set because we
	 * don't want to wander too far into the weeds; this should be
	 * sufficient to tell if something has gone terribly wrong with loading.
	 */
	for (c = 'A'; c <= 'Z'; c++) {
		lc = (c - 'A') + 'a';
		T_QUIET;
		T_ASSERT_EQ(lc, towlower(c), "tolower %c", c);
	}

	for (c = 0xc0; c <= 0xd6; c++) {
		lc = (c - 0xc0) + 0xe0;
		T_QUIET;
		T_ASSERT_EQ(lc, towlower(c), "tolower 0x%x", c);
	}
}

T_DECL(ctype_mapupper, "Check LC_CTYPE:mapupper table functionality")
{
	wchar_t c, uc;

	T_ASSERT_EQ_STR("en_US.UTF-8", setlocale(LC_CTYPE, "en_US.UTF-8"), NULL);

	for (c = 'a'; c <= 'z'; c++) {
		uc = (c - 'a') + 'A';
		T_QUIET;
		T_ASSERT_EQ(uc, towupper(c), "toupper %c", c);
	}

	for (c = 0xe0; c <= 0xf6; c++) {
		uc = (c - 0xe0) + 0xc0;
		T_QUIET;
		T_ASSERT_EQ(uc, towupper(c), "toupper 0x%x", c);
	}
}
#endif	/* TARGET_OS_OSX */
