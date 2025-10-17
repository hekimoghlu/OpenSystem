/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 17, 2023.
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
#include "../../btree/FreeBSD/bt_extern.h"

int	 __rec_close(DB *);
int	 __rec_delete(const DB *, const DBT *, u_int);
int	 __rec_dleaf(BTREE *, PAGE *, u_int32_t);
int	 __rec_fd(const DB *);
int	 __rec_fmap(BTREE *, recno_t);
int	 __rec_fout(BTREE *);
int	 __rec_fpipe(BTREE *, recno_t);
int	 __rec_get(const DB *, const DBT *, DBT *, u_int);
int	 __rec_iput(BTREE *, recno_t, const DBT *, u_int);
int	 __rec_put(const DB *dbp, DBT *, const DBT *, u_int);
int	 __rec_ret(BTREE *, EPG *, recno_t, DBT *, DBT *);
EPG	*__rec_search(BTREE *, recno_t, enum SRCHOP);
int	 __rec_seq(const DB *, DBT *, DBT *, u_int);
int	 __rec_sync(const DB *, u_int);
int	 __rec_vmap(BTREE *, recno_t);
int	 __rec_vout(BTREE *);
int	 __rec_vpipe(BTREE *, recno_t);
