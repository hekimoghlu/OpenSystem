/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 20, 2025.
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
struct date {
	int y;	/* year */
	int m;	/* month */
	int d;	/* day */
};

struct date	*easterg(int _year, struct date *_dt);
struct date	*easterog(int _year, struct date *_dt);
struct date	*easteroj(int _year, struct date *_dt);
struct date	*gdate(int _nd, struct date *_dt);
struct date	*jdate(int _nd, struct date *_dt);
int		 ndaysg(struct date *_dt);
int		 ndaysj(struct date *_dt);
int		 week(int _nd, int *_year);
int		 weekday(int _nd);
