/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 11, 2025.
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
/* $Id$ */

#ifndef HEIMDAL_KRB5_PAC_PLUGIN_H
#define HEIMDAL_KRB5_PAC_PLUGIN_H 1

#include <krb5.h>

/*
 * The PAC generate function should allocate a krb5_pac using
 * krb5_pac_init and fill in the PAC structure for the principal using
 * krb5_pac_add_buffer.
 *
 * The PAC verify function should verify all components in the PAC
 * using krb5_pac_get_types and krb5_pac_get_buffer for all types.
 *
 * Check client access function check if the client is authorized.
 */

struct hdb_entry_ex;
struct krb5_kdc_configuration;

typedef krb5_error_code
(*krb5plugin_windc_pac_generate)(void *, krb5_context,
				 struct hdb_entry_ex *, krb5_pac *);

typedef krb5_error_code
(*krb5plugin_windc_pac_verify)(void *, krb5_context,
			       const krb5_principal, /* new ticket client */
			       const krb5_principal, /* delegation proxy */
			       struct hdb_entry_ex *,/* client */
			       struct hdb_entry_ex *,/* server */
			       struct hdb_entry_ex *,/* krbtgt */
			       krb5_pac *);

typedef krb5_error_code
(*krb5plugin_windc_client_access)(
	void *, krb5_context,
	struct krb5_kdc_configuration *config,
	struct hdb_entry_ex *, const char *, 
	struct hdb_entry_ex *, const char *, 
	KDC_REQ *, METHOD_DATA *);


#define KRB5_WINDC_PLUGIN_MINOR			6
#define KRB5_WINDC_PLUGING_MINOR KRB5_WINDC_PLUGIN_MINOR

typedef struct krb5plugin_windc_ftable {
    int			minor_version;
    krb5_error_code	(*init)(krb5_context, void **);
    void		(*fini)(void *);
    krb5plugin_windc_pac_generate	pac_generate;
    krb5plugin_windc_pac_verify		pac_verify;
    krb5plugin_windc_client_access	client_access;
} krb5plugin_windc_ftable;

#endif /* HEIMDAL_KRB5_PAC_PLUGIN_H */

