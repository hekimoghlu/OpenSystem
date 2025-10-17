/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 16, 2024.
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
/*
 * This is considered an INTERNAL interface at this time.
 *
 * Some work is needed before exporting it:
 *
 * + Documentation.
 * + Sample code.
 * + Test cases (preferably automated testing under "make check").
 * + Hook into TGS exchange too; will change API.
 * + Examine memory management issues, especially for Windows; may
 *   change API.
 *
 * Other changes that would be nice to have, but not necessarily
 * before making this interface public:
 *
 * + Library support for AD-IF-RELEVANT and similar wrappers.  (We can
 *   make the plugin construct them if it wants them.)
 * + KDC could combine/optimize wrapped AD elements provided by
 *   multiple plugins, e.g., two IF-RELEVANT sequences could be
 *   merged.  (The preauth plugin API also has this bug, we're going
 *   to need a general fix.)
 */

#ifndef KRB5_AUTHDATA_PLUGIN_H_INCLUDED
#define KRB5_AUTHDATA_PLUGIN_H_INCLUDED
#include <Kerberos/krb5.h>

/*
 * While arguments of these types are passed-in, for the most part a
 * authorization data module can treat them as opaque.  If we need
 * keying data, we can ask for it directly.
 */
struct _krb5_db_entry_new;

/*
 * The function table / structure which an authdata server module must export as
 * "authdata_server_0".  NOTE: replace "0" with "1" for the type and
 * variable names if this gets picked up by upstream.  If the interfaces work
 * correctly, future versions of the table will add either more callbacks or
 * more arguments to callbacks, and in both cases we'll be able to wrap the v0
 * functions.
 */
/* extern krb5plugin_authdata_ftable_v0 authdata_server_0; */
typedef struct krb5plugin_authdata_ftable_v0 {
    /* Not-usually-visible name. */
    char *name;

    /*
     * Per-plugin initialization/cleanup.  The init function is called
     * by the KDC when the plugin is loaded, and the fini function is
     * called before the plugin is unloaded.  Both are optional.
     */
    krb5_error_code (*init_proc)(krb5_context, void **);
    void (*fini_proc)(krb5_context, void *);
    /*
     * Actual authorization data handling function.  If this field
     * holds a null pointer, this mechanism will be skipped, and the
     * init/fini functions will not be run.
     *
     * This function should only modify the field
     * enc_tkt_reply->authorization_data.  All other values should be
     * considered inputs only.  And, it should *modify* the field, not
     * overwrite it and assume that there are no other authdata
     * plugins in use.
     *
     * Memory management: authorization_data is a malloc-allocated,
     * null-terminated sequence of malloc-allocated pointers to
     * authorization data structures.  This plugin code currently
     * assumes the libraries, KDC, and plugin all use the same malloc
     * pool, which may be a problem if/when we get the KDC code
     * running on Windows.
     *
     * If this function returns a non-zero error code, a message
     * is logged, but no other action is taken.  Other authdata
     * plugins will be called, and a response will be sent to the
     * client (barring other problems).
     */
    krb5_error_code (*authdata_proc)(krb5_context,
				     struct _krb5_db_entry_new *client,
				     krb5_data *req_pkt,
				     krb5_kdc_req *request,
				     krb5_enc_tkt_part *enc_tkt_reply);
} krb5plugin_authdata_ftable_v0;
#endif /* KRB5_AUTHDATA_PLUGIN_H_INCLUDED */
