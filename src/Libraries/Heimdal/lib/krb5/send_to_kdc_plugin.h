/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 18, 2023.
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

#ifndef HEIMDAL_KRB5_SEND_TO_KDC_PLUGIN_H
#define HEIMDAL_KRB5_SEND_TO_KDC_PLUGIN_H 1

#include <krb5.h>

#define KRB5_PLUGIN_SEND_TO_KDC "send_to_kdc"

#define KRB5_PLUGIN_SEND_TO_KDC_VERSION_0 0
#define KRB5_PLUGIN_SEND_TO_KDC_VERSION_2 2
#define KRB5_PLUGIN_SEND_TO_KDC_VERSION KRB5_PLUGIN_SEND_TO_KDC_VERSION_2

typedef krb5_error_code
(*krb5plugin_send_to_kdc_func)(krb5_context,
			       void *,
			       krb5_krbhst_info *,
			       time_t timeout,
			       const krb5_data *,
			       krb5_data *);
typedef krb5_error_code
(*krb5plugin_send_to_realm_func)(krb5_context,
				 void *,
				 krb5_const_realm,
				 time_t timeout,
				 const krb5_data *,
				 krb5_data *);


typedef struct krb5plugin_send_to_kdc_ftable {
    int			minor_version;
    krb5_error_code	(*init)(krb5_context, void **);
    void		(*fini)(void *);
    krb5plugin_send_to_kdc_func send_to_kdc;
    krb5plugin_send_to_realm_func send_to_realm; /* added in version 2 */
} krb5plugin_send_to_kdc_ftable;

#endif /* HEIMDAL_KRB5_SEND_TO_KDC_PLUGIN_H */
