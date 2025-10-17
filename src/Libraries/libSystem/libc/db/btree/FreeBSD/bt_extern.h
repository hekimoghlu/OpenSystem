/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 26, 2021.
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
int	 __bt_close(DB *);
int	 __bt_cmp(BTREE *, const DBT *, EPG *);
int	 __bt_crsrdel(BTREE *, EPGNO *);
int	 __bt_defcmp(const DBT *, const DBT *);
size_t	 __bt_defpfx(const DBT *, const DBT *);
int	 __bt_delete(const DB *, const DBT *, u_int);
int	 __bt_dleaf(BTREE *, const DBT *, PAGE *, u_int);
int	 __bt_fd(const DB *);
int	 __bt_free(BTREE *, PAGE *);
int	 __bt_get(const DB *, const DBT *, DBT *, u_int);
PAGE	*__bt_new(BTREE *, pgno_t *);
void	 __bt_pgin(void *, pgno_t, void *);
void	 __bt_pgout(void *, pgno_t, void *);
int	 __bt_push(BTREE *, pgno_t, int);
int	 __bt_put(const DB *dbp, DBT *, const DBT *, u_int);
int	 __bt_ret(BTREE *, EPG *, DBT *, DBT *, DBT *, DBT *, int);
EPG	*__bt_search(BTREE *, const DBT *, int *);
int	 __bt_seq(const DB *, DBT *, DBT *, u_int);
void	 __bt_setcur(BTREE *, pgno_t, u_int);
int	 __bt_split(BTREE *, PAGE *,
	    const DBT *, const DBT *, int, size_t, u_int32_t);
int	 __bt_sync(const DB *, u_int);

int	 __ovfl_delete(BTREE *, void *);
int	 __ovfl_get(BTREE *, void *, size_t *, void **, size_t *);
int	 __ovfl_put(BTREE *, const DBT *, pgno_t *);

#ifdef DEBUG
void	 __bt_dnpage(DB *, pgno_t);
void	 __bt_dpage(PAGE *);
void	 __bt_dump(DB *);
#endif
#ifdef STATISTICS
void	 __bt_stat(DB *);
#endif
