/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 9, 2022.
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
#ifndef _GS2_TOKEN_H_
#define _GS2_TOKEN_H_ 1

#include <config.h>

#include <GSS/gssapi.h>

#ifndef KRB5_HEIMDAL
#ifdef HAVE_GSSAPI_GSSAPI_EXT_H
#include <gssapi/gssapi_ext.h>
#endif
#endif

#ifndef HAVE_GSS_DECAPSULATE_TOKEN
OM_uint32
gs2_decapsulate_token(const gss_buffer_t input_token,
                      const gss_OID token_oid,
                      gss_buffer_t output_token);
#define gss_decapsulate_token gs2_decapsulate_token
#endif

#ifndef HAVE_GSS_ENCAPSULATE_TOKEN
OM_uint32
gs2_encapsulate_token(const gss_buffer_t input_token,
                      const gss_OID token_oid,
                      gss_buffer_t output_token);
#define gss_encapsulate_token gs2_encapsulate_token
#endif

#ifndef HAVE_GSS_OID_EQUAL
int
gs2_oid_equal(const gss_OID o1, const gss_OID o2);
#define gss_oid_equal gs2_oid_equal
#endif

#endif /* _GS2_TOKEN_H_ */
