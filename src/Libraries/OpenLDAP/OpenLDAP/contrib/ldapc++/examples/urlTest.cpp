/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 1, 2021.
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
 * Copyright 2008-2011 The OpenLDAP Foundation, All Rights Reserved.
 * COPYING RESTRICTIONS APPLY, see COPYRIGHT file
 */

#include <LDAPUrl.h>
#include <LDAPException.h>
#include <cstdlib>
#include <iostream>

int main(int argc, char *argv[]) {
    if ( argc != 2 ) {
        std::cout << argc << std::endl;
        std::cout << "urlTest <ldap-URI>" << std::endl;
        exit(1);
    }
    std::string uristr = argv[1];
    try {
        LDAPUrl url(uristr);
        std::cout << "Host: " << url.getHost() << std::endl;
        std::cout << "Port: " << url.getPort() << std::endl;
        std::cout << "BaseDN: " << url.getDN() << std::endl;
        std::cout << "Scope: " << url.getScope() << std::endl;
        StringList attrs = url.getAttrs();
        std::cout << "Attrs: " << std::endl;
        StringList::const_iterator i = attrs.begin();
        for( ; i != attrs.end(); i++ ) {
            std::cout << "    " << *i << std::endl;
        }
        std::cout << "Filter: " << url.getFilter() << std::endl;
        std::cout << "Setting new BaseDN" << std::endl;
        url.setDN("o=Beispiel, c=DE");
        std::cout << "Url: " << url.getURLString() << std::endl;
    } catch (LDAPUrlException e) {
        std::cout << e.getCode() << std::endl;
        std::cout << e.getErrorMessage() << std::endl;
        std::cout << e.getAdditionalInfo() << std::endl;
    }

}
