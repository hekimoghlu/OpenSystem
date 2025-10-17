/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 12, 2021.
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
#pragma prototyped
/*
 * Glenn Fowler
 * AT&T Research
 *
 * time_t conversion support
 */

#include <tm.h>

/*
 * use one of the 14 equivalent calendar years to determine
 * daylight savings time for future years beyond the range
 * of the local system (via tmxtm())
 */

static const short equiv[] =
{
	2006, 2012,
	2001, 2024,
	2002, 2008,
	2003, 2020,
	2009, 2004,
	2010, 2016,
	2005, 2000,
};

/*
 * return the circa 2000 equivalent calendar year for tm
 */

int
tmequiv(Tm_t* tm)
{
	return tm->tm_year < (2038 - 1900) ? (tm->tm_year + 1900) : equiv[tm->tm_wday + tmisleapyear(tm->tm_year)];
}
