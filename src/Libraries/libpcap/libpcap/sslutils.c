/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 2, 2025.
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
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef HAVE_OPENSSL
#include <stdlib.h>

#include "portability.h"

#include "sslutils.h"

static const char *ssl_keyfile = "";   //!< file containing the private key in PEM format
static const char *ssl_certfile = "";  //!< file containing the server's certificate in PEM format
static const char *ssl_rootfile = "";  //!< file containing the list of CAs trusted by the client
// TODO: a way to set ssl_rootfile from the command line, or an envvar?

// TODO: lock?
static SSL_CTX *ctx;

void ssl_set_certfile(const char *certfile)
{
	ssl_certfile = certfile;
}

void ssl_set_keyfile(const char *keyfile)
{
	ssl_keyfile = keyfile;
}

int ssl_init_once(int is_server, int enable_compression, char *errbuf, size_t errbuflen)
{
	static int inited = 0;
	if (inited) return 0;

	SSL_library_init();
	SSL_load_error_strings();
	OpenSSL_add_ssl_algorithms();
	if (enable_compression)
		SSL_COMP_get_compression_methods();

	SSL_METHOD const *meth =
	    is_server ? SSLv23_server_method() : SSLv23_client_method();
	ctx = SSL_CTX_new(meth);
	if (! ctx)
	{
		snprintf(errbuf, errbuflen, "Cannot get a new SSL context: %s", ERR_error_string(ERR_get_error(), NULL));
		goto die;
	}

	SSL_CTX_set_mode(ctx, SSL_MODE_AUTO_RETRY);

	if (is_server)
	{
		char const *certfile = ssl_certfile[0] ? ssl_certfile : "cert.pem";
		if (1 != SSL_CTX_use_certificate_file(ctx, certfile, SSL_FILETYPE_PEM))
		{
			snprintf(errbuf, errbuflen, "Cannot read certificate file %s: %s", certfile, ERR_error_string(ERR_get_error(), NULL));
			goto die;
		}

		char const *keyfile = ssl_keyfile[0] ? ssl_keyfile : "key.pem";
		if (1 != SSL_CTX_use_PrivateKey_file(ctx, keyfile, SSL_FILETYPE_PEM))
		{
			snprintf(errbuf, errbuflen, "Cannot read private key file %s: %s", keyfile, ERR_error_string(ERR_get_error(), NULL));
			goto die;
		}
	}
	else
	{
		if (ssl_rootfile[0])
		{
			if (! SSL_CTX_load_verify_locations(ctx, ssl_rootfile, 0))
			{
				snprintf(errbuf, errbuflen, "Cannot read CA list from %s", ssl_rootfile);
				goto die;
			}
		}
		else
		{
			SSL_CTX_set_verify(ctx, SSL_VERIFY_NONE, NULL);
		}
	}

#if 0
	if (! RAND_load_file(RANDOM, 1024*1024))
	{
		snprintf(errbuf, errbuflen, "Cannot init random");
		goto die;
	}

	if (is_server)
	{
		SSL_CTX_set_session_id_context(ctx, (void *)&s_server_session_id_context, sizeof(s_server_session_id_context));
	}
#endif

	inited = 1;
	return 0;

die:
	return -1;
}

SSL *ssl_promotion(int is_server, SOCKET s, char *errbuf, size_t errbuflen)
{
	if (ssl_init_once(is_server, 1, errbuf, errbuflen) < 0) {
		return NULL;
	}

	SSL *ssl = SSL_new(ctx); // TODO: also a DTLS context
	SSL_set_fd(ssl, (int)s);

	if (is_server) {
		if (SSL_accept(ssl) <= 0) {
			snprintf(errbuf, errbuflen, "SSL_accept(): %s",
					ERR_error_string(ERR_get_error(), NULL));
			return NULL;
		}
	} else {
		if (SSL_connect(ssl) <= 0) {
			snprintf(errbuf, errbuflen, "SSL_connect(): %s",
					ERR_error_string(ERR_get_error(), NULL));
			return NULL;
		}
	}

	return ssl;
}

// Finish using an SSL handle; shut down the connection and free the
// handle.
void ssl_finish(SSL *ssl)
{
	//
	// We won't be using this again, so we can just send the
	// shutdown alert and free up the handle, and have our
	// caller close the socket.
	//
	// XXX - presumably, if the connection is shut down on
	// our side, either our peer won't have a problem sending
	// their shutdown alert or will not treat such a problem
	// as an error.  If this causes errors to be reported,
	// fix that as appropriate.
	//
	SSL_shutdown(ssl);
	SSL_free(ssl);
}

// Same return value as sock_send:
// 0 on OK, -1 on error but closed connection (-2).
int ssl_send(SSL *ssl, char const *buffer, int size, char *errbuf, size_t errbuflen)
{
	int status = SSL_write(ssl, buffer, size);
	if (status > 0)
	{
		// "SSL_write() will only return with success, when the complete contents (...) has been written."
		return 0;
	}
	else
	{
		int ssl_err = SSL_get_error(ssl, status); // TODO: does it pop the error?
		if (ssl_err == SSL_ERROR_ZERO_RETURN)
		{
			return -2;
		}
		else if (ssl_err == SSL_ERROR_SYSCALL)
		{
#ifndef _WIN32
			if (errno == ECONNRESET || errno == EPIPE) return -2;
#endif
		}
		snprintf(errbuf, errbuflen, "SSL_write(): %s",
		    ERR_error_string(ERR_get_error(), NULL));
		return -1;
	}
}

// Returns the number of bytes read, or -1 on syserror, or -2 on SSL error.
int ssl_recv(SSL *ssl, char *buffer, int size, char *errbuf, size_t errbuflen)
{
	int status = SSL_read(ssl, buffer, size);
	if (status <= 0)
	{
		int ssl_err = SSL_get_error(ssl, status);
		if (ssl_err == SSL_ERROR_ZERO_RETURN)
		{
			return 0;
		}
		else if (ssl_err == SSL_ERROR_SYSCALL)
		{
			return -1;
		}
		else
		{
			// Should not happen
			snprintf(errbuf, errbuflen, "SSL_read(): %s",
			    ERR_error_string(ERR_get_error(), NULL));
			return -2;
		}
	}
	else
	{
		return status;
	}
}

#endif // HAVE_OPENSSL
