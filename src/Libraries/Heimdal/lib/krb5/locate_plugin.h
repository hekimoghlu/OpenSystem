/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 19, 2022.
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

#ifndef HEIMDAL_KRB5_LOCATE_PLUGIN_H
#define HEIMDAL_KRB5_LOCATE_PLUGIN_H 1

#define KRB5_PLUGIN_LOCATE "service_locator"
#define KRB5_PLUGIN_LOCATE_VERSION 1
#define KRB5_PLUGIN_LOCATE_VERSION_0 0
#define KRB5_PLUGIN_LOCATE_VERSION_1 1
#define KRB5_PLUGIN_LOCATE_VERSION_2 2
#define KRB5_PLUGIN_LOCATE_VERSION_3 3

enum locate_service_type {
    locate_service_kdc = 1,
    locate_service_master_kdc,
    locate_service_kadmin,
    locate_service_krb524,
    locate_service_kpasswd
};

typedef krb5_error_code
(*krb5plugin_service_locate_lookup) (void *, unsigned long, enum locate_service_type,
				     const char *, int, int,
				     int (*)(void *,int,struct sockaddr *),
				     void *);

typedef krb5_error_code
(*krb5plugin_service_locate_lookup_host_string) (void *, unsigned long, enum locate_service_type,
				     const char *, int, int,
				     int (*)(void *,const char *),
				     void *);

#define KRB5_PLF_ALLOW_HOMEDIR	    1

typedef krb5_error_code
(*krb5plugin_service_locate_lookup_old) (void *, enum locate_service_type,
				     const char *, int, int,
				     int (*)(void *,int,struct sockaddr *),
				     void *);


typedef struct krb5plugin_service_locate_ftable {
    int			minor_version;
    krb5_error_code	(*init)(krb5_context, void **);
    void		(*fini)(void *);
    krb5plugin_service_locate_lookup_old old_lookup;
    krb5plugin_service_locate_lookup lookup; /* version 2 */
    krb5plugin_service_locate_lookup_host_string lookup_host_string; /* version 3 */
} krb5plugin_service_locate_ftable;

#endif /* HEIMDAL_KRB5_LOCATE_PLUGIN_H */

