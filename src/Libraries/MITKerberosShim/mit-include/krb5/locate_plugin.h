/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 25, 2023.
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
#ifndef KRB5_LOCATE_PLUGIN_H_INCLUDED
#define KRB5_LOCATE_PLUGIN_H_INCLUDED
#include <krb5/krb5.h>
#include <Availability.h>

enum locate_service_type {
    locate_service_kdc = 1,
    locate_service_master_kdc,
    locate_service_kadmin,
    locate_service_krb524,
    locate_service_kpasswd
} __API_DEPRECATED("Use GSS.framework", macos(10.0, 10.8)) __API_UNAVAILABLE(iosmac);

typedef struct __API_DEPRECATED("Use GSS.framework", macos(10.0, 10.8)) __API_UNAVAILABLE(iosmac) krb5plugin_service_locate_ftable {
    int minor_version;		/* currently 0 */
    /* Per-context setup and teardown.  Returned void* blob is
       private to the plugin.  */
    krb5_error_code (*init)(krb5_context, void **);
    void (*fini)(void *);
    /* Callback function returns non-zero if the plugin function
       should quit and return; this may be because of an error, or may
       indicate we've already contacted the service, whatever.  The
       lookup function should only return an error if it detects a
       problem, not if the callback function tells it to quit.  */
    krb5_error_code (*lookup)(void *,
			      enum locate_service_type svc, const char *realm,
			      int socktype, int family,
			      int (*cbfunc)(void *,int,struct sockaddr *),
			      void *cbdata);
} krb5plugin_service_locate_ftable __API_DEPRECATED("Use GSS.framework", macos(10.0, 10.8)) __API_UNAVAILABLE(iosmac);
/* extern krb5plugin_service_locate_ftable service_locator; */
#endif
