/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 18, 2024.
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
/* System library. */

#include <sys_defs.h>

#ifdef USE_TLS

/* Utility library. */

#include <msg.h>

/* Global library. */

#include <mail_params.h>

/* TLS library. */

#define TLS_INTERNAL
#include <tls.h>

/* tls_set_ca_certificate_info - load Certification Authority certificates */

int     tls_set_ca_certificate_info(SSL_CTX *ctx, const char *CAfile,
				            const char *CApath)
{
    if (*CAfile == 0)
	CAfile = 0;
    if (*CApath == 0)
	CApath = 0;

#define CA_PATH_FMT "%s%s%s"
#define CA_PATH_ARGS(var, nextvar) \
	var ? #var "=\"" : "", \
	var ? var : "", \
	var ? (nextvar ? "\", " : "\"") : ""

    if (CAfile || CApath) {
	if (!SSL_CTX_load_verify_locations(ctx, CAfile, CApath)) {
	    msg_info("cannot load Certification Authority data, "
		     CA_PATH_FMT CA_PATH_FMT ": disabling TLS support",
		     CA_PATH_ARGS(CAfile, CApath),
		     CA_PATH_ARGS(CApath, 0));
	    tls_print_errors();
	    return (-1);
	}
	if (var_tls_append_def_CA && !SSL_CTX_set_default_verify_paths(ctx)) {
	    msg_info("cannot set default OpenSSL certificate verification "
		     "paths: disabling TLS support");
	    tls_print_errors();
	    return (-1);
	}
    }
    return (0);
}

/* set_cert_stuff - specify certificate and key information */

static int set_cert_stuff(SSL_CTX *ctx, const char *cert_type,
			          const char *cert_file,
			          const char *key_file)
{

    /*
     * We need both the private key (in key_file) and the public key
     * certificate (in cert_file). Both may specify the same file.
     * 
     * Code adapted from OpenSSL apps/s_cb.c.
     */
    ERR_clear_error();
    if (SSL_CTX_use_certificate_chain_file(ctx, cert_file) <= 0) {
	msg_warn("cannot get %s certificate from file \"%s\": "
		 "disabling TLS support", cert_type, cert_file);
	tls_print_errors();
	return (0);
    }
    if (SSL_CTX_use_PrivateKey_file(ctx, key_file, SSL_FILETYPE_PEM) <= 0) {
	msg_warn("cannot get %s private key from file \"%s\": "
		 "disabling TLS support", cert_type, key_file);
	tls_print_errors();
	return (0);
    }

    /*
     * Sanity check.
     */
    if (!SSL_CTX_check_private_key(ctx)) {
	msg_warn("%s private key in %s does not match public key in %s: "
		 "disabling TLS support", cert_type, key_file, cert_file);
	return (0);
    }
    return (1);
}

/* tls_set_my_certificate_key_info - load client or server certificates/keys */

int     tls_set_my_certificate_key_info(SSL_CTX *ctx,
					        const char *cert_file,
					        const char *key_file,
					        const char *dcert_file,
					        const char *dkey_file,
					        const char *eccert_file,
					        const char *eckey_file)
{

    /*
     * Lack of certificates is fine so long as we are prepared to use
     * anonymous ciphers.
     */
    if (*cert_file && !set_cert_stuff(ctx, "RSA", cert_file, key_file))
	return (-1);			/* logged */
    if (*dcert_file && !set_cert_stuff(ctx, "DSA", dcert_file, dkey_file))
	return (-1);				/* logged */
#if OPENSSL_VERSION_NUMBER >= 0x1000000fL && !defined(OPENSSL_NO_ECDH)
    if (*eccert_file && !set_cert_stuff(ctx, "ECDSA", eccert_file, eckey_file))
	return (-1);				/* logged */
#else
    if (*eccert_file)
	msg_warn("ECDSA not supported. Ignoring ECDSA certificate file \"%s\"",
		 eccert_file);
#endif
    return (0);
}

#endif
