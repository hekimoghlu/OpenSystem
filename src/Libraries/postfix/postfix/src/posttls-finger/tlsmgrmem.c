/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 3, 2022.
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
#include <sys_defs.h>

#ifdef USE_TLS
#include <htable.h>
#include <vstring.h>
#include <tls_mgr.h>

#include "tlsmgrmem.h"

static HTABLE *tls_cache;
static int cache_enabled = 1;
static int cache_count;
static int cache_hits;
typedef void (*free_func) (void *);
static free_func free_value = (free_func) vstring_free;

void    tlsmgrmem_disable(void)
{
    cache_enabled = 0;
}

void    tlsmgrmem_flush(void)
{
    if (!tls_cache)
	return;
    htable_free(tls_cache, free_value);
}

void    tlsmgrmem_status(int *enabled, int *count, int *hits)
{
    if (enabled)
	*enabled = cache_enabled;
    if (count)
	*count = cache_count;
    if (hits)
	*hits = cache_hits;
}

/* tls_mgr_* - Local cache and stubs that do not talk to the TLS manager */

int     tls_mgr_seed(VSTRING *buf, int len)
{
    return (TLS_MGR_STAT_OK);
}

int     tls_mgr_policy(const char *unused_type, int *cachable, int *timeout)
{
    if (cache_enabled && tls_cache == 0)
	tls_cache = htable_create(1);
    *cachable = cache_enabled;
    *timeout = TLS_SESSION_LIFEMIN;
    return (TLS_MGR_STAT_OK);
}

int     tls_mgr_lookup(const char *unused_type, const char *key, VSTRING *buf)
{
    VSTRING *s;

    if (tls_cache == 0)
	return TLS_MGR_STAT_ERR;

    if ((s = (VSTRING *) htable_find(tls_cache, key)) == 0)
	return TLS_MGR_STAT_ERR;

    vstring_memcpy(buf, vstring_str(s), VSTRING_LEN(s));

    ++cache_hits;
    return (TLS_MGR_STAT_OK);
}

int     tls_mgr_update(const char *unused_type, const char *key,
		               const char *buf, ssize_t len)
{
    HTABLE_INFO *ent;
    VSTRING *s;

    if (tls_cache == 0)
	return TLS_MGR_STAT_ERR;

    if ((ent = htable_locate(tls_cache, key)) == 0) {
	s = vstring_alloc(len);
	ent = htable_enter(tls_cache, key, (void *) s);
    } else {
	s = (VSTRING *) ent->value;
    }
    vstring_memcpy(s, buf, len);

    ++cache_count;
    return (TLS_MGR_STAT_OK);
}

int     tls_mgr_delete(const char *unused_type, const char *key)
{
    if (tls_cache == 0)
	return TLS_MGR_STAT_ERR;

    if (htable_locate(tls_cache, key)) {
	htable_delete(tls_cache, key, free_value);
	--cache_count;
    }
    return (TLS_MGR_STAT_OK);
}

#endif
