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
/* This work is part of OpenLDAP Software <http://www.openldap.org/>.
 *
 * Copyright 2007-2011 The OpenLDAP Foundation.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted only as authorized by the OpenLDAP
 * Public License.
 *
 * A copy of this license is available in the file LICENSE in the
 * top-level directory of the distribution or, alternatively, at
 * <http://www.OpenLDAP.org/license.html>.
 */
/* ACKNOWLEDGEMENTS:
 * This work was initially developed by Brian Candler for inclusion
 * in OpenLDAP Software.
 */

#ifndef _PROTO_SOCK_H
#define _PROTO_SOCK_H

LDAP_BEGIN_DECL

extern BI_init		sock_back_initialize;

extern BI_open		sock_back_open;
extern BI_close		sock_back_close;
extern BI_destroy	sock_back_destroy;

extern BI_db_init	sock_back_db_init;
extern BI_db_destroy	sock_back_db_destroy;

extern BI_op_bind	sock_back_bind;
extern BI_op_unbind	sock_back_unbind;
extern BI_op_search	sock_back_search;
extern BI_op_compare	sock_back_compare;
extern BI_op_modify	sock_back_modify;
extern BI_op_modrdn	sock_back_modrdn;
extern BI_op_add	sock_back_add;
extern BI_op_delete	sock_back_delete;

extern int sock_back_init_cf( BackendInfo *bi );

LDAP_END_DECL

#endif /* _PROTO_SOCK_H */
