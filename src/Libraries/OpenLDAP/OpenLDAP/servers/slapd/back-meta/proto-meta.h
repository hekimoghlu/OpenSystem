/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 11, 2025.
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
 * Copyright 1999-2011 The OpenLDAP Foundation.
 * Portions Copyright 2001-2003 Pierangelo Masarati.
 * Portions Copyright 1999-2003 Howard Chu.
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
 * This work was initially developed by the Howard Chu for inclusion
 * in OpenLDAP Software and subsequently enhanced by Pierangelo
 * Masarati.
 */

#ifndef PROTO_META_H
#define PROTO_META_H

LDAP_BEGIN_DECL

extern BI_init			meta_back_initialize;

extern BI_open			meta_back_open;
extern BI_close			meta_back_close;
extern BI_destroy		meta_back_destroy;

extern BI_db_init		meta_back_db_init;
extern BI_db_open		meta_back_db_open;
extern BI_db_destroy		meta_back_db_destroy;
extern BI_db_config		meta_back_db_config;

extern BI_op_bind		meta_back_bind;
extern BI_op_search		meta_back_search;
extern BI_op_compare		meta_back_compare;
extern BI_op_modify		meta_back_modify;
extern BI_op_modrdn		meta_back_modrdn;
extern BI_op_add		meta_back_add;
extern BI_op_delete		meta_back_delete;
extern BI_op_abandon		meta_back_abandon;

extern BI_connection_destroy	meta_back_conn_destroy;

LDAP_END_DECL

#endif /* PROTO_META_H */
