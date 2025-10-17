/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 3, 2023.
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
#ifndef	_xa_ext_h_
#define	_xa_ext_h_

#if defined(__cplusplus)
extern "C" {
#endif

int __xa_get_txn __P((ENV *, DB_TXN **, int));
int __db_xa_create __P((DB *));
int __db_rmid_to_env __P((int, ENV **));
int __db_xid_to_txn __P((ENV *, XID *, roff_t *));
int __db_map_rmid __P((int, ENV *));
int __db_unmap_rmid __P((int));
int __db_map_xid __P((ENV *, XID *, TXN_DETAIL *));
void __db_unmap_xid __P((ENV *, XID *, size_t));

#if defined(__cplusplus)
}
#endif
#endif /* !_xa_ext_h_ */
