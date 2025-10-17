/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 1, 2021.
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
/*
 * NICachePrivate.h
 */

#ifndef _S_NICACHE_PRIVATE_H
#define _S_NICACHE_PRIVATE_H

#include <stdint.h>

PLCacheEntry_t *PLCacheEntry_create(ni_proplist pl);
void		PLCacheEntry_free(PLCacheEntry_t * ent);

void		PLCache_init(PLCache_t * cache);
void		PLCache_free(PLCache_t * cache);
int		PLCache_count(PLCache_t * c);
boolean_t	PLCache_read(PLCache_t * cache, const char * filename);
boolean_t	PLCache_write(PLCache_t * cache, const char * filename);
void		PLCache_add(PLCache_t * cache, PLCacheEntry_t * entry);
void		PLCache_append(PLCache_t * cache, PLCacheEntry_t * entry);
void		PLCache_remove(PLCache_t * cache, PLCacheEntry_t * entry);
void		PLCache_set_max(PLCache_t * c, int m);
PLCacheEntry_t *PLCache_lookup_prop(PLCache_t * PLCache, 
				    char * prop, char * value, boolean_t make_head);
PLCacheEntry_t *PLCache_lookup_hw(PLCache_t * PLCache, 
				  uint8_t hwtype, void * hwaddr, int hwlen,
				  NICacheFunc_t * func, void * arg,
				  struct in_addr * client_ip,
				  boolean_t * has_binding);
PLCacheEntry_t *PLCache_lookup_identifier(PLCache_t * PLCache, 
					  char * idstr, 
					  NICacheFunc_t * func, void * arg,
					  struct in_addr * client_ip,
					  boolean_t * has_binding);
PLCacheEntry_t *PLCache_lookup_ip(PLCache_t * PLCache, struct in_addr iaddr);
void		PLCache_make_head(PLCache_t * cache, PLCacheEntry_t * entry);
void		PLCache_print(PLCache_t * cache);

#endif /* _S_NICACHE_PRIVATE_H */
