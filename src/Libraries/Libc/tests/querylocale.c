/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 28, 2024.
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
#include <locale.h>
#include <xlocale.h>

#include <darwintest.h>
#include <TargetConditionals.h>

#ifndef nitems
#define	nitems(x)	(sizeof((x)) / sizeof((x)[0]))
#endif

/*
 * We don't install the necessary localedata on embedded platforms to be able to
 * usefully run this tests, so just limit it to macOS.
 */
T_DECL(querylocale_names, "Check that querylocale() returns names",
    T_META_ENABLED(TARGET_OS_OSX))
{
	const char *lcat;
	int mask = LC_ALL_MASK;

	T_ASSERT_EQ_STR("en_US.UTF-8", setlocale(LC_ALL, "en_US.UTF-8"), NULL);

	while (mask != 0) {
		lcat = querylocale(mask, NULL);

		T_ASSERT_EQ_STR("en_US.UTF-8", lcat, NULL);
		mask &= ~(1 << (ffs(mask) - 1));
	}
}

T_DECL(querylocale_newlocale_names,
    "Check that querylocale() returns names for newlocale() locales",
    T_META_ENABLED(TARGET_OS_OSX))
{
	const char *lcat;
	locale_t nlocale;
	int mask = LC_ALL_MASK;

	nlocale = newlocale(LC_ALL_MASK, "en_US.UTF-8", NULL);

	while (mask != 0) {
		lcat = querylocale(mask, nlocale);

		T_ASSERT_EQ_STR("en_US.UTF-8", lcat, NULL);
		mask &= ~(1 << (ffs(mask) - 1));
	}

	freelocale(nlocale);
}

/* We expect alphabetical order. */
static int order_mapping[] = {
	[0] = LC_COLLATE,
	[1] = LC_CTYPE,
	[2] = LC_MESSAGES,
	[3] = LC_MONETARY,
	[4] = LC_NUMERIC,
	[5] = LC_TIME,
};

T_DECL(querylocale_order, "Check the querylocale() mask mapping",
    T_META_ENABLED(TARGET_OS_OSX))
{
	const char *lcat;
	int cat;

	for (size_t i = 0; i < nitems(order_mapping); i++) {
		cat = order_mapping[i];

		T_QUIET;
		T_ASSERT_EQ_STR("C", setlocale(cat, NULL), NULL);

		T_ASSERT_EQ_STR("en_US.UTF-8", setlocale(cat, "en_US.UTF-8"),
		    NULL);

		lcat = querylocale(1 << i, NULL);
		T_ASSERT_EQ_STR("en_US.UTF-8", lcat, NULL);

		T_ASSERT_EQ_STR("C", setlocale(cat, "C"), NULL);
	}
}
