/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 2, 2023.
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
#include <stdio.h>
#include <string.h>

#include <CNIOBoringSSL_asn1.h>
#include <CNIOBoringSSL_conf.h>
#include <CNIOBoringSSL_err.h>
#include <CNIOBoringSSL_mem.h>
#include <CNIOBoringSSL_obj.h>
#include <CNIOBoringSSL_x509.h>

#include "../internal.h"
#include "ext_dat.h"


static char *i2s_ASN1_IA5STRING(const X509V3_EXT_METHOD *method, void *ext) {
  const ASN1_IA5STRING *ia5 = reinterpret_cast<const ASN1_IA5STRING *>(ext);
  char *tmp;
  if (!ia5 || !ia5->length) {
    return NULL;
  }
  if (!(tmp = reinterpret_cast<char *>(OPENSSL_malloc(ia5->length + 1)))) {
    return NULL;
  }
  OPENSSL_memcpy(tmp, ia5->data, ia5->length);
  tmp[ia5->length] = 0;
  return tmp;
}

static void *s2i_ASN1_IA5STRING(const X509V3_EXT_METHOD *method,
                                const X509V3_CTX *ctx, const char *str) {
  ASN1_IA5STRING *ia5;
  if (!str) {
    OPENSSL_PUT_ERROR(X509V3, X509V3_R_INVALID_NULL_ARGUMENT);
    return NULL;
  }
  if (!(ia5 = ASN1_IA5STRING_new())) {
    goto err;
  }
  if (!ASN1_STRING_set(ia5, str, strlen(str))) {
    ASN1_IA5STRING_free(ia5);
    goto err;
  }
  return ia5;
err:
  return NULL;
}

#define EXT_IA5STRING(nid)                                                 \
  {                                                                        \
    nid, 0, ASN1_ITEM_ref(ASN1_IA5STRING), 0, 0, 0, 0, i2s_ASN1_IA5STRING, \
        s2i_ASN1_IA5STRING, 0, 0, 0, 0, NULL                               \
  }

#define EXT_END \
  { -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }

const X509V3_EXT_METHOD v3_ns_ia5_list[] = {
    EXT_IA5STRING(NID_netscape_base_url),
    EXT_IA5STRING(NID_netscape_revocation_url),
    EXT_IA5STRING(NID_netscape_ca_revocation_url),
    EXT_IA5STRING(NID_netscape_renewal_url),
    EXT_IA5STRING(NID_netscape_ca_policy_url),
    EXT_IA5STRING(NID_netscape_ssl_server_name),
    EXT_IA5STRING(NID_netscape_comment),
    EXT_END};
