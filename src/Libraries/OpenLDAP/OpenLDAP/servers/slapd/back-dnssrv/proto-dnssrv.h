/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 19, 2024.
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
 * Copyright 2000-2011 The OpenLDAP Foundation.
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
 * This work was originally developed by Kurt D. Zeilenga for inclusion
 * in OpenLDAP Software.
 */

#ifndef PROTO_DNSSRV_H
#define PROTO_DNSSRV_H

LDAP_BEGIN_DECL

extern BI_init			dnssrv_back_initialize;

extern BI_open			dnssrv_back_open;
extern BI_close			dnssrv_back_close;
extern BI_destroy		dnssrv_back_destroy;

extern BI_db_init		dnssrv_back_db_init;
extern BI_db_destroy		dnssrv_back_db_destroy;
extern BI_db_config		dnssrv_back_db_config;

extern BI_op_bind		dnssrv_back_bind;
extern BI_op_search		dnssrv_back_search;
extern BI_op_compare		dnssrv_back_compare;

extern BI_chk_referrals		dnssrv_back_referrals;

extern AttributeDescription	*ad_dc;
extern AttributeDescription	*ad_associatedDomain;

LDAP_END_DECL

#endif /* PROTO_DNSSRV_H */
