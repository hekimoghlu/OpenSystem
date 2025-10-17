/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 31, 2023.
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
#include "krb5_locl.h"

/**
 * Set the absolute time that the caller knows the kdc has so the
 * kerberos library can calculate the relative diffrence beteen the
 * KDC time and local system time.
 *
 * @param context Kerberos 5 context.
 * @param sec The applications new of "now" in seconds
 * @param usec The applications new of "now" in micro seconds

 * @return Kerberos 5 error code, see krb5_get_error_message().
 *
 * @ingroup krb5
 */

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_set_real_time (krb5_context context,
		    krb5_timestamp sec,
		    int32_t usec)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);

    context->kdc_sec_offset = (uint32_t)(sec - tv.tv_sec);

    /**
     * If the caller passes in a negative usec, its assumed to be
     * unknown and the function will use the current time usec.
     */
    if (usec >= 0) {
	context->kdc_usec_offset = usec - tv.tv_usec;

	if (context->kdc_usec_offset < 0) {
	    context->kdc_sec_offset--;
	    context->kdc_usec_offset += 1000000;
	}
    } else
	context->kdc_usec_offset = tv.tv_usec;

    return 0;
}

/*
 * return ``corrected'' time in `timeret'.
 */

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_timeofday (krb5_context context,
		krb5_timestamp *timeret)
{
    *timeret = time(NULL);
    if (context)
	*timeret += context->kdc_sec_offset;
    return 0;
}

/*
 * like gettimeofday but with time correction to the KDC
 */

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_us_timeofday (krb5_context context,
		   krb5_timestamp *sec,
		   int32_t *usec)
{
    struct timeval tv;

    gettimeofday (&tv, NULL);

    *sec  = tv.tv_sec;
    if (context)
	*sec += context->kdc_sec_offset;
    *usec = tv.tv_usec;
    return 0;
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_format_time(krb5_context context, time_t t,
		 char *s, size_t len, krb5_boolean include_time)
{
    struct tm *tm;
    if(context->log_utc)
	tm = gmtime (&t);
    else
	tm = localtime(&t);
    if(tm == NULL ||
       strftime(s, len, include_time ? context->time_fmt : context->date_fmt, tm) == 0)
	snprintf(s, len, "%ld", (long)t);
    return 0;
}

KRB5_LIB_FUNCTION krb5_error_code KRB5_LIB_CALL
krb5_string_to_deltat(const char *string, krb5_deltat *deltat)
{
    if((*deltat = parse_time(string, "s")) == -1)
	return KRB5_DELTAT_BADFORMAT;
    return 0;
}

krb5_deltat
krb5_time_abs(krb5_deltat t1, krb5_deltat t2)
{
    krb5_deltat t = t1 - t2;
    if (t < 0)
	return -t;
    return t;
}
