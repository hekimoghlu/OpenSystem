/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 29, 2022.
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

#include <iostream>
#include <string>
#include "LDAPAsynConnection.h"
#include "TlsOptions.h"

int main( int argc, char* argv[]){
    if ( argc != 4 ){
        std::cerr << "usage: " << argv[0] << " <ldap-uri> <cacertfile> <cacertdir>" << std::endl;
        return(-1);
    }
    std::string uri(argv[1]);
    std::string cacert(argv[2]);
    std::string cadir(argv[3]);
    TlsOptions tls;
    std::cout << "Current global settings:" << std::endl;
    std::cout << "    CaCertfile: " << tls.getStringOption( TlsOptions::CACERTFILE) << std::endl;
    std::cout << "    CaCertDir: " << tls.getStringOption( TlsOptions::CACERTDIR ) << std::endl;
    std::cout << "    Require Cert: " << tls.getIntOption( TlsOptions::REQUIRE_CERT ) << std::endl;
    std::cout << "Applying new settings:" << std::endl;
    tls.setOption( TlsOptions::CACERTFILE, cacert );
    tls.setOption( TlsOptions::REQUIRE_CERT, TlsOptions::DEMAND );
    std::cout << "    CaCertfile: " << tls.getStringOption( TlsOptions::CACERTFILE ) << std::endl;
    std::cout << "    Require Cert: " << tls.getIntOption( TlsOptions::REQUIRE_CERT ) << std::endl;

    try {
        // 1. connect using global options
        LDAPAsynConnection l(uri);
        try {
            l.start_tls();
            std::cout << "StartTLS successful." << std::endl;
            l.unbind();
        } catch ( LDAPException e ) {
            std::cerr << e << std::endl;
        }

        // 2. connect using connection specific option
        LDAPAsynConnection l1(uri);
        tls=l1.getTlsOptions();
        std::cout << "Current connection specific settings:" << std::endl;
        std::cout << "    CaCertfile: " << tls.getStringOption( TlsOptions::CACERTFILE) << std::endl;
        std::cout << "    CaCertDir: " << tls.getStringOption( TlsOptions::CACERTDIR ) << std::endl;
        std::cout << "    Require Cert: " << tls.getIntOption( TlsOptions::REQUIRE_CERT ) << std::endl;
        std::cout << "Applying new settings:" << std::endl;
        tls.setOption( TlsOptions::CACERTDIR, cadir );
        tls.setOption( TlsOptions::REQUIRE_CERT, TlsOptions::DEMAND );
        std::cout << "    CaCertDir: " << tls.getStringOption( TlsOptions::CACERTDIR ) << std::endl;
        std::cout << "    Require Cert: " << tls.getIntOption( TlsOptions::REQUIRE_CERT ) << std::endl;
        try {
            l1.start_tls();
            std::cout << "StartTLS successful." << std::endl;
            l1.unbind();
        } catch ( LDAPException e ) {
            std::cerr << e << std::endl;
        }

        // 3. and once again using the globals
        try {
            LDAPAsynConnection l2(uri);
            TlsOptions tls2;
            std::cout << "Current global settings:" << std::endl;
            std::cout << "    CaCertfile: " << tls2.getStringOption( TlsOptions::CACERTFILE) << std::endl;
            std::cout << "    CaCertDir: " << tls2.getStringOption( TlsOptions::CACERTDIR ) << std::endl;
            std::cout << "    Require Cert: " << tls2.getIntOption( TlsOptions::REQUIRE_CERT ) << std::endl;
            l2.start_tls();
            std::cout << "StartTLS successful." << std::endl;
            l2.unbind();
        } catch ( LDAPException e ) {
            std::cerr << e << std::endl;
        }
    } catch ( LDAPException e ) {
        std::cerr << e << std::endl;
    }
}
