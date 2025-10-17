/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 30, 2022.
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
#include "der_locl.h"

#define ASN1_MAX_YEAR	2000

static int
is_leap(unsigned y)
{
    y += 1900;
    return (y % 4) == 0 && ((y % 100) != 0 || (y % 400) == 0);
}

static const unsigned ndays[2][12] ={
    {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31},
    {31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31}};

/*
 * This is a simplifed version of timegm(3) that doesn't accept out of
 * bound values that timegm(3) normally accepts but those are not
 * valid in asn1 encodings.
 */

time_t
_der_timegm (struct tm *tm)
{
  time_t res = 0;
  int i;

  /*
   * See comment in _der_gmtime
   */
  if (tm->tm_year > ASN1_MAX_YEAR)
      return 0;

  if (tm->tm_year < 0)
      return -1;
  if (tm->tm_mon < 0 || tm->tm_mon > 11)
      return -1;
  if (tm->tm_mday < 1 || tm->tm_mday > (int)ndays[is_leap(tm->tm_year)][tm->tm_mon])
      return -1;
  if (tm->tm_hour < 0 || tm->tm_hour > 23)
      return -1;
  if (tm->tm_min < 0 || tm->tm_min > 59)
      return -1;
  if (tm->tm_sec < 0 || tm->tm_sec > 60)
      return -1;

  for (i = 70; i < tm->tm_year; ++i)
    res += is_leap(i) ? 366 : 365;

  for (i = 0; i < tm->tm_mon; ++i)
    res += ndays[is_leap(tm->tm_year)][i];
  res += tm->tm_mday - 1;
  res *= 24;
  res += tm->tm_hour;
  res *= 60;
  res += tm->tm_min;
  res *= 60;
  res += tm->tm_sec;
  return res;
}

struct tm *
_der_gmtime(time_t t, struct tm *tm)
{
    time_t secday = t % (3600 * 24);
    time_t days = t / (3600 * 24);

    memset(tm, 0, sizeof(*tm));

    tm->tm_sec = secday % 60;
    tm->tm_min = (secday % 3600) / 60;
    tm->tm_hour = (int)(secday / 3600);

    /*
     * Refuse to calculate time ~ 2000 years into the future, this is
     * not possible for systems where time_t is a int32_t, however,
     * when time_t is a int64_t, that can happen, and this becomes a
     * denial of sevice.
     */
    if (days > (ASN1_MAX_YEAR * 365))
	return NULL;

    tm->tm_year = 70;
    while(1) {
	time_t dayinyear = (is_leap(tm->tm_year) ? 366 : 365);
	if (days < dayinyear)
	    break;
	tm->tm_year += 1;
	days -= dayinyear;
    }
    tm->tm_mon = 0;

    while (1) {
	time_t daysinmonth = ndays[is_leap(tm->tm_year)][tm->tm_mon];
	if (days < daysinmonth)
	    break;
	days -= daysinmonth;
	tm->tm_mon++;
    }
    tm->tm_mday = (int)(days + 1);

    return tm;
}
