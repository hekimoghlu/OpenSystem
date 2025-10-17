/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 26, 2022.
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
/* System library. */

#include <sys_defs.h>

/* Utility library. */

#include <msg.h>
#include <dict.h>

/* Global library. */

#include <maps.h>

/* Application-specific. */

#include <postscreen.h>

 /*
  * Monitor time-critical operations.
  * 
  * XXX Averaging support was added during a stable release candidate, so it
  * provides only the absolute minimum necessary. A complete implementation
  * should maintain separate statistics for each table, and it should not
  * complain when the access latency is less than the time between accesses.
  */
#define PSC_GET_TIME_BEFORE_LOOKUP { \
    struct timeval _before, _after; \
    DELTA_TIME _delta; \
    double _new_delta_ms; \
    GETTIMEOFDAY(&_before);

#define PSC_DELTA_MS(d) ((d).dt_sec * 1000.0 + (d).dt_usec / 1000.0)

#define PSC_AVERAGE(new, old)	(0.1 * (new) + 0.9 * (old))

#ifndef PSC_THRESHOLD_MS
#define PSC_THRESHOLD_MS	100	/* nag if latency > 100ms */
#endif

#ifndef PSC_WARN_LOCKOUT_S
#define PSC_WARN_LOCKOUT_S	60	/* don't nag for 60s */
#endif

 /*
  * Shared warning lock, so that we don't spam the logfile when the system
  * becomes slow.
  */
static time_t psc_last_warn = 0;

#define PSC_CHECK_TIME_AFTER_LOOKUP(table, action, average) \
    GETTIMEOFDAY(&_after); \
    PSC_CALC_DELTA(_delta, _after, _before); \
    _new_delta_ms = PSC_DELTA_MS(_delta); \
    if ((average = PSC_AVERAGE(_new_delta_ms, average)) > PSC_THRESHOLD_MS \
	&& psc_last_warn < _after.tv_sec - PSC_WARN_LOCKOUT_S) { \
        msg_warn("%s: %s %s average delay is %.0f ms", \
                 myname, (table), (action), average); \
	psc_last_warn = _after.tv_sec; \
    } \
}

/* psc_addr_match_list_match - time-critical address list lookup */

int     psc_addr_match_list_match(ADDR_MATCH_LIST *addr_list,
				          const char *addr_str)
{
    const char *myname = "psc_addr_match_list_match";
    int     result;
    static double latency_ms;

    PSC_GET_TIME_BEFORE_LOOKUP;
    result = addr_match_list_match(addr_list, addr_str);
    PSC_CHECK_TIME_AFTER_LOOKUP("address list", "lookup", latency_ms);
    return (result);
}

/* psc_cache_lookup - time-critical cache lookup */

const char *psc_cache_lookup(DICT_CACHE *cache, const char *key)
{
    const char *myname = "psc_cache_lookup";
    const char *result;
    static double latency_ms;

    PSC_GET_TIME_BEFORE_LOOKUP;
    result = dict_cache_lookup(cache, key);
    PSC_CHECK_TIME_AFTER_LOOKUP(dict_cache_name(cache), "lookup", latency_ms);
    return (result);
}

/* psc_cache_update - time-critical cache update */

void    psc_cache_update(DICT_CACHE *cache, const char *key, const char *value)
{
    const char *myname = "psc_cache_update";
    static double latency_ms;

    PSC_GET_TIME_BEFORE_LOOKUP;
    dict_cache_update(cache, key, value);
    PSC_CHECK_TIME_AFTER_LOOKUP(dict_cache_name(cache), "update", latency_ms);
}

/* psc_dict_get - time-critical table lookup */

const char *psc_dict_get(DICT *dict, const char *key)
{
    const char *myname = "psc_dict_get";
    const char *result;
    static double latency_ms;

    PSC_GET_TIME_BEFORE_LOOKUP;
    result = dict_get(dict, key);
    PSC_CHECK_TIME_AFTER_LOOKUP(dict->name, "lookup", latency_ms);
    return (result);
}

/* psc_maps_find - time-critical table lookup */

const char *psc_maps_find(MAPS *maps, const char *key, int flags)
{
    const char *myname = "psc_maps_find";
    const char *result;
    static double latency_ms;

    PSC_GET_TIME_BEFORE_LOOKUP;
    result = maps_find(maps, key, flags);
    PSC_CHECK_TIME_AFTER_LOOKUP(maps->title, "lookup", latency_ms);
    return (result);
}
