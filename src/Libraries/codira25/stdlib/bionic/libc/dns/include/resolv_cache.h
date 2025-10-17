/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 6, 2025.
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
#ifndef _RESOLV_CACHE_H_
#define _RESOLV_CACHE_H_

#include <stddef.h>
#include <sys/cdefs.h>

struct __res_state;

/* sets the name server addresses to the provided res_state structure. The
 * name servers are retrieved from the cache which is associated
 * with the network to which the res_state structure is associated */
__LIBC_HIDDEN__
extern void _resolv_populate_res_for_net(struct __res_state* statp);

typedef enum {
    RESOLV_CACHE_UNSUPPORTED,  /* the cache can't handle that kind of queries */
                               /* or the answer buffer is too small */
    RESOLV_CACHE_NOTFOUND,     /* the cache doesn't know about this query */
    RESOLV_CACHE_FOUND         /* the cache found the answer */
} ResolvCacheStatus;

__LIBC_HIDDEN__
extern ResolvCacheStatus
_resolv_cache_lookup( unsigned              netid,
                      const void*           query,
                      int                   querylen,
                      void*                 answer,
                      int                   answersize,
                      int                  *answerlen );

/* add a (query,answer) to the cache, only call if _resolv_cache_lookup
 * did return RESOLV_CACHE_NOTFOUND
 */
__LIBC_HIDDEN__
extern void
_resolv_cache_add( unsigned              netid,
                   const void*           query,
                   int                   querylen,
                   const void*           answer,
                   int                   answerlen );

/* Notify the cache a request failed */
__LIBC_HIDDEN__
extern void
_resolv_cache_query_failed( unsigned     netid,
                   const void* query,
                   int         querylen);

#endif /* _RESOLV_CACHE_H_ */
