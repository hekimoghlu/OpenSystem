/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 23, 2024.
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

// $OpenLDAP$
/*
 * Copyright 2010-2011 The OpenLDAP Foundation, All Rights Reserved.
 * COPYING RESTRICTIONS APPLY, see COPYRIGHT file
 */

#include "TlsOptions.h"
#include "LDAPException.h"

enum opttype {
    INT=0,
    STRING,
    OTHER
};

typedef struct tls_optmap {
    int optval;
    opttype type;
} tls_optmap_t;

static tls_optmap_t optmap[] = {
    { LDAP_OPT_X_TLS_CACERTFILE, STRING },
    { LDAP_OPT_X_TLS_CACERTDIR, STRING },
    { LDAP_OPT_X_TLS_CERTFILE, STRING },
    { LDAP_OPT_X_TLS_KEYFILE, STRING },
    { LDAP_OPT_X_TLS_REQUIRE_CERT, INT },
    { LDAP_OPT_X_TLS_PROTOCOL_MIN, INT },
    { LDAP_OPT_X_TLS_CIPHER_SUITE, STRING },
    { LDAP_OPT_X_TLS_RANDOM_FILE, STRING },
    { LDAP_OPT_X_TLS_CRLCHECK, INT },
    { LDAP_OPT_X_TLS_DHFILE, STRING },
    { LDAP_OPT_X_TLS_NEWCTX, INT }
};
#if 0 /* not implemented currently */
        static const int TLS_CRLFILE /* GNUtls only */
        static const int TLS_SSL_CTX  /* OpenSSL SSL* */
        static const int TLS_CONNECT_CB
        static const int TLS_CONNECT_ARG
#endif 

static void checkOpt( TlsOptions::tls_option opt, opttype type ) {
    if ( opt < TlsOptions::CACERTFILE || opt >= TlsOptions::LASTOPT ){
        throw( LDAPException( LDAP_PARAM_ERROR, "unknown Option" ) );
    }

    if ( optmap[opt].type != type ){
        throw( LDAPException( LDAP_PARAM_ERROR, "not a string option" ) );
    }
}

TlsOptions::TlsOptions() : m_ld(NULL) {}

TlsOptions::TlsOptions( LDAP* ld ): m_ld(ld) { }

void TlsOptions::setOption( tls_option opt, const std::string& value ) const {
    checkOpt(opt, STRING);
    this->setOption( opt, value.empty() ? NULL : (void*) value.c_str() );
}

void TlsOptions::setOption( tls_option opt, int value ) const {
    checkOpt(opt, INT);
    this->setOption( opt, (void*) &value);
}

void TlsOptions::setOption( tls_option opt, void *value ) const {
    int ret = ldap_set_option( m_ld, optmap[opt].optval, value);
    if ( ret != LDAP_OPT_SUCCESS )
    {
        if ( ret != LDAP_OPT_ERROR ){
            throw( LDAPException( ret ));
        } else {
            throw( LDAPException( LDAP_PARAM_ERROR, "error while setting TLS option" ) );
        }
    }
    this->newCtx();
}

void TlsOptions::getOption( tls_option opt, void* value ) const {
    int ret = ldap_get_option( m_ld, optmap[opt].optval, value);
    if ( ret != LDAP_OPT_SUCCESS )
    {
        if ( ret != LDAP_OPT_ERROR ){
            throw( LDAPException( ret ));
        } else {
            throw( LDAPException( LDAP_PARAM_ERROR, "error while reading TLS option" ) );
        }
    }
}

int TlsOptions::getIntOption( tls_option opt ) const {
    int value;
    checkOpt(opt, INT);
    ldap_get_option( m_ld, optmap[opt].optval, (void*) &value);
    return value;
}

std::string TlsOptions::getStringOption( tls_option opt ) const {
    char *value;
    checkOpt(opt, STRING);
    ldap_get_option( m_ld, optmap[opt].optval, (void*) &value);
    std::string strval;
    if (value)
    {
        strval=std::string(value);
        ldap_memfree(value);
    }
    return strval;
}

void TlsOptions::newCtx() const {
    int val = 0;
    int ret = ldap_set_option( m_ld, LDAP_OPT_X_TLS_NEWCTX, &val);
    if ( ret != LDAP_OPT_SUCCESS )
    {
        if ( ret != LDAP_OPT_ERROR ){
            throw( LDAPException( ret ));
        } else {
            throw( LDAPException( LDAP_LOCAL_ERROR, "error while renewing TLS context" ) );
        }
    }
}
