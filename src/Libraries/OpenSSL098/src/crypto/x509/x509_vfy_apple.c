/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 21, 2024.
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
#include <stdint.h>
#include <inttypes.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>
#include <TrustEvaluationAgent/TrustEvaluationAgent.h>
#include <syslog.h>

#include "cryptlib.h"
#include "x509_vfy_apple.h"

#define TEA_might_correct_error(err) (err == X509_V_ERR_UNABLE_TO_GET_ISSUER_CERT_LOCALLY || err == X509_V_ERR_UNABLE_TO_GET_ISSUER_CERT || err == X509_V_ERR_SELF_SIGNED_CERT_IN_CHAIN)

/*
 * Please see comment in x509_vfy_apple.h
 */
int
X509_verify_cert(X509_STORE_CTX *ctx)
{
	TEAResult		ret = kTEAResultCertNotTrusted;
	TEACertificateChainRef	inputChain = NULL;
	TEACertificateChainRef	outputChain = NULL;
	__block uint64_t	certCount = 0;
	uint64_t		certLastIndex = 0;
	uint64_t		i = 0;
	int			error = 0;
	TEAParams		params = { 0 };


	if (ctx == NULL || ctx->cert == NULL)
        return kTEAResultErrorOccured;

	/* Try OpenSSL, if we get a local certificate issue verify against trusted roots */
	ret = X509_verify_cert_orig(ctx);

	/* Verify TEA is enabled and should be used. */
	if (0 != X509_TEA_is_enabled() &&
		ret != 1 && TEA_might_correct_error(ctx->error)) {

		/* Verify that the certificate chain exists, otherwise make it. */
		if (ctx->chain == NULL && (ctx->chain = sk_X509_new_null()) == NULL) {
			TEALogDebug("Could not create the certificate chain");
			ret = kTEAResultCertNotTrusted;
			goto bail;
		}

		/* Verify chain depth */
		certLastIndex = sk_X509_num(ctx->untrusted);
		if (certLastIndex > ctx->param->depth) {
			TEALogDebug("Pruning certificate chain to %" PRIu64, certLastIndex);
			certLastIndex = ctx->param->depth;
		}

		inputChain = TEACertificateChainCreate();
		if (inputChain == NULL) {
			TEALogDebug("Certificate chain creation failed");
			goto bail;
		}

		unsigned char *asn1_cert_data = NULL;
		int asn1_cert_len = i2d_X509(ctx->cert, &asn1_cert_data);
		error = TEACertificateChainAddCert(inputChain, asn1_cert_data, asn1_cert_len);
		// TEACertificateChainAddCert made a copy of the ASN.1 data, so we get to free ours here
		OPENSSL_free(asn1_cert_data);
		if (error) {
			TEALogDebug("An error occured while inserting the certificate into the chain");
			goto bail;
		}

		for (i = 0; i < certLastIndex; ++i) {
			X509	*t = sk_X509_value(ctx->untrusted, i);

			asn1_cert_data = NULL;
			asn1_cert_len = i2d_X509(t, &asn1_cert_data);
			error = TEACertificateChainAddCert(inputChain, asn1_cert_data, asn1_cert_len);
			// TEACertificateChainAddCert made a copy of the ASN.1 data, so we get to free ours here
			OPENSSL_free(asn1_cert_data);
			if (error) {
				TEALogDebug("An error occured while inserting an untrusted certificate into the chain");
				goto bail;
			}
		}

		// We put ASN.1 encoded X509 on the CertificateChain, so we don't call TEACertificateChainSetEncodingHandler
		
		params.purpose = ctx->param->purpose;
		if (ctx->param->flags & X509_V_FLAG_USE_CHECK_TIME)
			params.time = ctx->param->check_time;

		outputChain = TEAVerifyCert(inputChain, &params);

		TEACertificateChainRelease(inputChain);
		inputChain = NULL;

		if (outputChain == NULL) {
			TEALogDebug("TEAVerifyCert() returned NULL.");
			goto bail;
		}

		/* Empty the context chain */
		for (i = 0; i < sk_X509_num(ctx->chain); ++i)
			sk_X509_pop(ctx->chain);

		error = TEACertificateChainGetCerts(outputChain, ^(const TEACertificateRef cert) {
			const unsigned char	*ptr = TEACertificateGetData(cert);
			X509			*c = NULL;

			if (certCount++ > certLastIndex)
				return 0;

			c = d2i_X509(NULL, &ptr, TEACertificateGetSize(cert));
			if (c == NULL) {
				TEALogDebug("Could not parse certificate");
				return 1;
			}

			if (!sk_X509_push(ctx->chain, c)) {
				TEALogDebug("Could not insert certificate into the chain");
				return 1;
			}

			return 0;
		});
		if (error) {
			TEALogDebug("An error occured while deserializing the trusted certificate chain");
			ret = kTEAResultCertNotTrusted;
			goto bail;
		}

		TEACertificateChainRelease(outputChain);
		outputChain = NULL;

		/* Fixup context data */
		ctx->current_cert   = sk_X509_value(ctx->chain, 0);
		ctx->current_issuer = sk_X509_value(ctx->chain, sk_X509_num(ctx->chain) - 1);
		ctx->error_depth = 0;
		ctx->error = 0;
		X509_get_pubkey_parameters(NULL, ctx->chain);

		ret = kTEAResultCertTrusted;
	}

bail:
	if (inputChain) {
		TEACertificateChainRelease(inputChain);
		inputChain = NULL;
	}
	if (outputChain) {
		TEACertificateChainRelease(outputChain);
		outputChain = NULL;
	}
	return ret;
}

#pragma mark Trust Evaluation Agent

/* -1: not set
 *  0: set to false
 *  1: set to true
 */
static int tea_enabled = -1;

void
X509_TEA_set_state(int change)
{
	tea_enabled = (change) ? 1 : 0;
}

int
X509_TEA_is_enabled()
{
	if (tea_enabled < 0)
		tea_enabled = (NULL == getenv(X509_TEA_ENV_DISABLE));

	return tea_enabled != 0;
}
