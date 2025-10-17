/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 21, 2023.
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
#import "mylog.h"
#import "cfutil.h"
#import "symbol_scope.h"
#import "PvDInfoContext.h"
#import "IPHPvDInfoRequestUtil.h"

PRIVATE_EXTERN void
PvDInfoContextFlush(PvDInfoContext * ret_context, bool persist_failure)
{
	CFStringRef pvdid_save = NULL;
	bool status_ok = ret_context->status_ok;

	if (persist_failure && !status_ok) {
		pvdid_save = ret_context->pvdid;
	} else {
		my_CFRelease(&ret_context->pvdid);
	}
	my_CFRelease(&ret_context->ipv6_prefixes);
	my_CFRelease(&ret_context->additional_info);
	my_CFRelease(&ret_context->last_fetched_date);
	my_CFRelease(&ret_context->effective_expiration_date);
	bzero(ret_context, sizeof(*ret_context));
	if (persist_failure && !status_ok) {
		ret_context->pvdid = pvdid_save;
		ret_context->status_ok = status_ok;
	}
}

PRIVATE_EXTERN CFStringRef
PvDInfoContextGetPvDID(PvDInfoContext * ret_context)
{
	return (ret_context->pvdid);
}

PRIVATE_EXTERN void
PvDInfoContextSetPvDID(PvDInfoContext * ret_context, CFStringRef pvdid)
{
	if (pvdid != NULL) {
		CFRetain(pvdid);
	}
	my_CFRelease(&ret_context->pvdid);
	ret_context->pvdid = pvdid;
	return;
}

PRIVATE_EXTERN CFArrayRef
PvDInfoContextGetPrefixes(PvDInfoContext * ret_context)
{
	return (ret_context->ipv6_prefixes);
}

PRIVATE_EXTERN void
PvDInfoContextSetPrefixes(PvDInfoContext * ret_context, CFArrayRef prefixes)
{
	if (prefixes != NULL) {
		CFRetain(prefixes);
	}
	my_CFRelease(&ret_context->ipv6_prefixes);
	ret_context->ipv6_prefixes = prefixes;
	return;
}

PRIVATE_EXTERN const char *
PvDInfoContextGetInterfaceName(PvDInfoContext * ret_context)
{
	return (ret_context->if_name);
}

PRIVATE_EXTERN void
PvDInfoContextSetInterfaceName(PvDInfoContext * ret_context,
			       const char * if_name)
{
	ret_context->if_name = if_name;
	return;
}

PRIVATE_EXTERN uint16_t
PvDInfoContextGetSequenceNumber(PvDInfoContext * ret_context)
{
	return (ret_context->sequence_nr);
}

PRIVATE_EXTERN void
PvDInfoContextSetSequenceNumber(PvDInfoContext * ret_context, uint16_t seqnr)
{
	ret_context->sequence_nr = seqnr;
	return;
}

PRIVATE_EXTERN bool
PvDInfoContextIsOk(PvDInfoContext * ret_context)
{
	return (ret_context->status_ok);
}

PRIVATE_EXTERN void
PvDInfoContextSetIsOk(PvDInfoContext * ret_context, bool ok)
{
	ret_context->status_ok = ok;
	return;
}

PRIVATE_EXTERN CFDictionaryRef
PvDInfoContextGetAdditionalInformation(PvDInfoContext * ret_context)
{
	return (ret_context->additional_info);
}

PRIVATE_EXTERN void
PvDInfoContextSetAdditionalInformation(PvDInfoContext * ret_context,
				       CFDictionaryRef additional_info)
{
	if (additional_info != NULL) {
		CFRetain(additional_info);
	}
	my_CFRelease(&ret_context->additional_info);
	ret_context->additional_info = additional_info;
	return;
}

PRIVATE_EXTERN bool
PvDInfoContextCanRefetch(PvDInfoContext * ret_context)
{
	return (((CFDateGetAbsoluteTime(ret_context->last_fetched_date)
		  + kPvDInfoClientRefetchSamePvDIDMinWaitSeconds)
		 < CFAbsoluteTimeGetCurrent()));
}

PRIVATE_EXTERN bool
PvDInfoContextFetchAllowed(PvDInfoContext * ret_context)
{
	return (ret_context->status_ok);
}

PRIVATE_EXTERN void
PvDInfoContextSetLastFetchedDateToNow(PvDInfoContext * ret_context)
{
	my_CFRelease(&ret_context->last_fetched_date);
	ret_context->last_fetched_date
	= CFDateCreate(NULL, CFAbsoluteTimeGetCurrent());;
	return;
}

INLINE CFDateFormatterRef
_date_formatter_create(void)
{
	CFLocaleRef locale = NULL;
	CFDateFormatterRef date_formatter = NULL;

	locale = CFLocaleCreate(NULL, kPvDInfoExpirationDateLocale);
	if (locale == NULL) {
		goto done;
	}
	date_formatter = CFDateFormatterCreate(NULL, locale,
					       kCFDateFormatterNoStyle,
					       kCFDateFormatterNoStyle);
	if (date_formatter == NULL) {
		goto done;
	}
	CFDateFormatterSetFormat(date_formatter, kPvDInfoExpirationDateFormat);

done:
	my_CFRelease(&locale);
	return (date_formatter);
}

INLINE CFDateRef
_expiration_date_and_string_create(CFStringRef expiration_str,
				   CFStringRef * ret_new_fetch_date_str)
{
	CFDateFormatterRef date_formatter = NULL;
	CFAbsoluteTime expiration_time = 0;
	CFAbsoluteTime now_time = 0;
	CFAbsoluteTime interval_start_time = 0;
	CFAbsoluteTime now_until_exp = 0;
	uint32_t randomization_domain = 0;
	uint64_t random_distance_from_interval_start = 0;
	CFAbsoluteTime new_fetch_time_seconds = 0;
	CFDateRef effective_expiration_date = NULL;

	if ((date_formatter = _date_formatter_create()) == NULL) {
		goto done;
	}
	if (!CFDateFormatterGetAbsoluteTimeFromString(date_formatter,
						      expiration_str,
						      NULL,
						      &expiration_time)) {
		goto done;
	}
	now_time = CFAbsoluteTimeGetCurrent();
	interval_start_time = (now_time + expiration_time)/2;
	/*
	 * (now)       (interval_start)     (expiration)
	 *   |<--------------->|<--------------->|
	 *                              ^
	 *      refetch somewhere here__|
	 */
	now_until_exp = (expiration_time - interval_start_time);
	randomization_domain = (uint32_t)((now_until_exp
					   > (CFAbsoluteTime)UINT32_MAX)
					  ? UINT32_MAX
					  : now_until_exp);
	random_distance_from_interval_start
	= (uint64_t)arc4random_uniform(randomization_domain);
	new_fetch_time_seconds
	= (interval_start_time
	   + (CFAbsoluteTime)random_distance_from_interval_start);
	effective_expiration_date = CFDateCreate(NULL, new_fetch_time_seconds);
	if (effective_expiration_date == NULL) {
		goto done;
	}
	*ret_new_fetch_date_str
	= CFDateFormatterCreateStringWithAbsoluteTime(NULL, date_formatter,
						      new_fetch_time_seconds);

done:
	my_CFRelease(&date_formatter);
	return (effective_expiration_date);
}

PRIVATE_EXTERN void
PvDInfoContextCalculateEffectiveExpiration(PvDInfoContext * ret_context)
{
	CFDictionaryRef additional_info_dict = NULL;
	CFStringRef expiration_str = NULL;
	CFStringRef new_fetch_date_str = NULL;
	CFDateRef effective_expiration_date = NULL;
	bool success = false;

	my_log(LOG_DEBUG, "%s", __func__);
	additional_info_dict = ret_context->additional_info;
	if (additional_info_dict == NULL) {
		goto done;
	}
	expiration_str
	= CFDictionaryGetValue(additional_info_dict,
			       kPvDInfoAdditionalInfoDictKeyExpires);
	effective_expiration_date
	= _expiration_date_and_string_create(expiration_str,
					     &new_fetch_date_str);
	if (effective_expiration_date == NULL) {
		goto done;
	}
	my_log(LOG_INFO,
	       "%s: PvD info with ID '%@' has effective expiration date '%@'",
	       __func__, ret_context->pvdid, new_fetch_date_str);
	my_CFRelease(&ret_context->effective_expiration_date);
	ret_context->effective_expiration_date = effective_expiration_date;
	success = true;

done:
	if (!success) {
		my_log(LOG_ERR, "%s: couldn't set expiration date "
		       "for pvdid '%@' with addinfo '%@'",
		       __func__, ret_context->pvdid, additional_info_dict);
		my_CFRelease(&ret_context->effective_expiration_date);
	}
	my_CFRelease(&new_fetch_date_str);
	return;
}

PRIVATE_EXTERN CFAbsoluteTime
PvDInfoContextGetEffectiveExpirationTime(PvDInfoContext * ret_context)
{
	return ((ret_context->effective_expiration_date != NULL)
		? CFDateGetAbsoluteTime(ret_context->effective_expiration_date)
		: 0);
}
