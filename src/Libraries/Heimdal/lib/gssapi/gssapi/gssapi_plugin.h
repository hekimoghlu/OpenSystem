/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 13, 2022.
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
#ifndef __GSSAPI_PLUGIN_H
#define __GSSAPI_PLUGIN_H 1

#define GSSAPI_PLUGIN "gssapi_plugin"

typedef gss_cred_id_t
(*gssapi_plugin_isc_replace_cred)(gss_name_t target, gss_OID mech, gss_cred_id_t original_cred, OM_uint32 flags);

/*
 * Flags passed in the flags argument to ->isc_replace_cred()
 */
#define GPT_IRC_F_SYSTEM_ONLY	1 /* system resource only, home directory access is no allowed */

/*
 * Flags defined by the plugin in gssapi_plugin_ftable
 */
#define GPT_SYSTEM_ONLY		1	/* plugin support GPT_IRC_F_SYSTEM_ONLY and friends */

/*
 * Plugin for GSSAPI 
 */

typedef struct gssapi_plugin_ftable {
    int			minor_version; /* support protocol: GSSAPI_PLUGIN_VERSION_N */
    krb5_error_code	(*init)(krb5_context, void **);
    void		(*fini)(void *);
    const char		*name;
    unsigned long	flags;
    gssapi_plugin_isc_replace_cred isc_replace_cred;
} gssapi_plugin_ftable;

#define GSSAPI_PLUGIN_VERSION_1 1

/* history of version changes:
 * version 0 (no supported) was missing flags argument to ->isc_replace_cred()
 */

#endif

