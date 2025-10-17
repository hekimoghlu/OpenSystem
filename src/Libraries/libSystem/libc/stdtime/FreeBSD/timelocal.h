/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 7, 2024.
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
#ifndef _TIMELOCAL_H_
#define	_TIMELOCAL_H_

#include <xlocale.h>

/*
 * Private header file for the strftime and strptime localization
 * stuff.
 */
struct lc_time_T {
	const char	*mon[12];
	const char	*month[12];
	const char	*wday[7];
	const char	*weekday[7];
	const char	*X_fmt;
	const char	*x_fmt;
	const char	*c_fmt;
	const char	*am;
	const char	*pm;
	const char	*date_fmt;
	const char	*alt_month[12];
	const char	*md_order;
	const char	*ampm_fmt;
};

struct lc_time_T *__get_current_time_locale(locale_t);
int	__time_load_locale(const char *, locale_t);

#endif /* !_TIMELOCAL_H_ */
