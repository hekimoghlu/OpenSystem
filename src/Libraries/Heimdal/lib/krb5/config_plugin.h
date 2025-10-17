/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 4, 2025.
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
#ifndef HEIMDAL_KRB5_CONFIG_PLUGIN_H
#define HEIMDAL_KRB5_CONFIG_PLUGIN_H 1

#include <krb5.h>

#define KRB5_PLUGIN_CONFIGURATION "krb5_configuration"

typedef krb5_error_code
(*krb5plugin_get_default_realm)(krb5_context, void *, void *, void (*)(krb5_context, void *, krb5_const_realm));
typedef krb5_error_code
(*krb5plugin_get_host_domain)(krb5_context, const char *, void *, void *, void (*)(krb5_context, void *, krb5_const_realm));

#define KRB5_PLUGIN_CONFIGURATION_VERSION_0	0
#define KRB5_PLUGIN_CONFIGURATION_VERSION_1	1

typedef struct krb5plugin_config_ftable {
    int			minor_version;
    krb5_error_code	(*init)(krb5_context, void **);
    void		(*fini)(void *);
    krb5plugin_get_default_realm get_default_realm;
    krb5plugin_get_host_domain get_host_domain;
} krb5plugin_config_ftable;

#endif /* HEIMDAL_KRB5_CONFIG_PLUGIN_H */
