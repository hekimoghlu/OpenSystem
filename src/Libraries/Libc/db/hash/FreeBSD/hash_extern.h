/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 29, 2024.
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
BUFHEAD	*__add_ovflpage(HTAB *, BUFHEAD *);
int	 __addel(HTAB *, BUFHEAD *, const DBT *, const DBT *);
int	 __big_delete(HTAB *, BUFHEAD *);
int	 __big_insert(HTAB *, BUFHEAD *, const DBT *, const DBT *);
int	 __big_keydata(HTAB *, BUFHEAD *, DBT *, DBT *, int);
int	 __big_return(HTAB *, BUFHEAD *, int, DBT *, int);
int	 __big_split(HTAB *, BUFHEAD *, BUFHEAD *, BUFHEAD *,
		int, u_int32_t, SPLIT_RETURN *);
int	 __buf_free(HTAB *, int, int);
void	 __buf_init(HTAB *, int);
u_int32_t	 __call_hash(HTAB *, char *, int);
int	 __delpair(HTAB *, BUFHEAD *, int);
int	 __expand_table(HTAB *);
int	 __find_bigpair(HTAB *, BUFHEAD *, int, char *, int);
u_int16_t	 __find_last_page(HTAB *, BUFHEAD **);
void	 __free_ovflpage(HTAB *, BUFHEAD *);
BUFHEAD	*__get_buf(HTAB *, u_int32_t, BUFHEAD *, int);
int	 __get_page(HTAB *, char *, u_int32_t, int, int, int);
int	 __ibitmap(HTAB *, int, int, int);
u_int32_t	 __log2(u_int32_t);
int	 __put_page(HTAB *, char *, u_int32_t, int, int);
void	 __reclaim_buf(HTAB *, BUFHEAD *);
int	 __split_page(HTAB *, u_int32_t, u_int32_t);

/* Default hash routine. */
extern u_int32_t (*__default_hash)(const void *, size_t);

#ifdef HASH_STATISTICS
extern int hash_accesses, hash_collisions, hash_expansions, hash_overflows;
#endif
