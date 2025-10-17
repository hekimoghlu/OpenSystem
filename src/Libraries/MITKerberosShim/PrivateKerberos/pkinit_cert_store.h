/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 14, 2022.
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
 * pkinit_cert_store.h - PKINIT certificate storage/retrieval utilities
 *
 * Created 26 May 2004 by Doug Mitchell at Apple.
 */
 
#ifndef	_PKINIT_CERT_STORE_H_
#define _PKINIT_CERT_STORE_H_

#include <krb5/krb5.h>

#ifdef  __cplusplus
extern "C" {
#endif

/*
 * Opaque reference to a machine-dependent representation of a certificate
 * which is capable of signing. On Mac OS X this is actually a SecIdentityRef.
 */
typedef void *krb5_pkinit_signing_cert_t;

/*
 * Opaque reference to a machine-dependent representation of a certificate.
 * On Mac OS X this is actually a SecCertificateRef.
 */
typedef void *krb5_pkinit_cert_t;

/* 
 * Opaque reference to a database in which PKINIT-related certificates are stored. 
 */
typedef void *krb5_pkinit_cert_db_t;

/*
 * Obtain signing cert for specified principal. On successful return, 
 * caller must eventually release the cert with krb5_pkinit_release_cert().
 *
 * Returns KRB5_PRINC_NOMATCH if client cert not found.
 */
krb5_error_code krb5_pkinit_get_client_cert(
    const char			*principal,     /* full principal string */
    krb5_pkinit_signing_cert_t	*client_cert);  /* RETURNED */
    
/* 
 * Determine if the specified client has a signing cert. Returns TRUE
 * if so, else returns FALSE.
 */
krb5_boolean krb5_pkinit_have_client_cert(
    const char			*principal);    /* full principal string */

/*
 * Store the specified certificate (or, more likely, some platform-dependent
 * reference to it) as the specified principal's signing cert. Passing
 * in NULL for the client_cert has the effect of deleting the relevant entry
 * in the cert storage.
 */
krb5_error_code krb5_pkinit_set_client_cert(
    const char			*principal,     /* full principal string */
    krb5_pkinit_cert_t	client_cert);

/* 
 * Obtain a reference to the client's cert database. Specify either principal
 * name or client_cert as obtained from krb5_pkinit_get_client_cert().
 */
krb5_error_code krb5_pkinit_get_client_cert_db(
    const char			*principal,	    /* optional, full principal string */
    krb5_pkinit_signing_cert_t	client_cert,	    /* optional, from krb5_pkinit_get_client_cert() */
    krb5_pkinit_cert_db_t	*client_cert_db);   /* RETURNED */

/*
 * Obtain the KDC signing cert, with optional CA and specific cert specifiers.
 * CAs and cert specifiers are in the form of DER-encoded issuerAndSerialNumbers.
 *
 * The client_spec argument is typically provided by the client as kdcPkId.
 *
 * If trusted_CAs and client_spec are NULL, a platform-dependent preferred 
 * KDC signing cert is returned, if one exists. 
 *
 * On successful return, caller must eventually release the cert with 
 * krb5_pkinit_release_cert(). Outside of an unusual test configuration this =
 *
 * Returns KRB5_PRINC_NOMATCH if KDC cert not found.
 *
 */
krb5_error_code krb5_pkinit_get_kdc_cert(
    krb5_ui_4			num_trusted_CAs,    /* sizeof *trusted_CAs */
    krb5_data			*trusted_CAs,	    /* optional */
    krb5_data			*client_spec,	    /* optional */
    krb5_pkinit_signing_cert_t	*kdc_cert);	    /* RETURNED */

/* 
 * Obtain a reference to the KDC's cert database.
 */
krb5_error_code krb5_pkinit_get_kdc_cert_db(
    krb5_pkinit_cert_db_t	*kdc_cert_db);	/* RETURNED */

/*
 * Release certificate references obtained via krb5_pkinit_get_client_cert() and
 * krb5_pkinit_get_kdc_cert().
 */
extern void krb5_pkinit_release_cert(
    krb5_pkinit_signing_cert_t   cert);
    
/*
 * Release database references obtained via krb5_pkinit_get_client_cert_db() and
 * krb5_pkinit_get_kdc_cert_db().
 */
extern void krb5_pkinit_release_cert_db(
    krb5_pkinit_cert_db_t	cert_db);
    
/* 
 * Obtain a mallocd C-string representation of a certificate's SHA1 digest. 
 * Only error is a NULL return indicating memory failure. 
 * Caller must free the returned string.
 */
char *krb5_pkinit_cert_hash_str(
    const krb5_data *cert);
    
#ifdef  __cplusplus
}
#endif

#endif  /* _PKINIT_CERT_STORE_H_ */
