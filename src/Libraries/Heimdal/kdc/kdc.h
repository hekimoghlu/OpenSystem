/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 21, 2024.
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
 * $Id$
 */

#ifndef __KDC_H__
#define __KDC_H__

#include <hdb.h>
#include <krb5.h>

enum krb5_kdc_trpolicy {
    TRPOLICY_ALWAYS_CHECK,
    TRPOLICY_ALLOW_PER_PRINCIPAL,
    TRPOLICY_ALWAYS_HONOUR_REQUEST
};

typedef struct krb5_kdc_configuration {
    krb5_boolean require_preauth; /* require preauth for all principals */
    time_t kdc_warn_pwexpire; /* time before expiration to print a warning */

    struct HDB **db;
    unsigned int num_db;

    krb5_boolean encode_as_rep_as_tgs_rep; /* bug compatibility */

    krb5_boolean as_use_strongest_session_key;
    krb5_boolean preauth_use_strongest_session_key;
    krb5_boolean tgs_use_strongest_session_key;
    krb5_boolean use_strongest_server_key;

    krb5_boolean check_ticket_addresses;
    krb5_boolean allow_null_ticket_addresses;
    krb5_boolean allow_anonymous;
    enum krb5_kdc_trpolicy trpolicy;

    krb5_boolean enable_pkinit;
    krb5_boolean pkinit_princ_in_cert;
    const char *pkinit_kdc_identity;
    const char *pkinit_kdc_anchors;
    const char *pkinit_kdc_friendly_name;
    const char *pkinit_kdc_ocsp_file;
    char **pkinit_kdc_cert_pool;
    char **pkinit_kdc_revoke;
    int pkinit_dh_min_bits;
    int pkinit_require_binding;
    int pkinit_allow_proxy_certs;

    krb5_log_facility *logf;

    int enable_digest;
    int digests_allowed;

    size_t max_datagram_reply_length;

    int enable_kx509;
    const char *kx509_template;
    const char *kx509_ca;

    char *lkdc_realm;

} krb5_kdc_configuration;

struct krb5_kdc_service {
    unsigned int flags;
#define KS_KRB5		1
    krb5_error_code (*process)(krb5_context context,
			       krb5_kdc_configuration *config,
			       krb5_data *req_buffer,
			       krb5_data *reply,
			       const char *from,
			       struct sockaddr *addr,
			       size_t max_reply_size,
			       int *claim);
};

#include <kdc-protos.h>

#endif
