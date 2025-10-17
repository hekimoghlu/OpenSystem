/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 19, 2023.
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
 * Copyright 2007-2011 The OpenLDAP Foundation, All Rights Reserved.
 * COPYING RESTRICTIONS APPLY, see COPYRIGHT file
 */

#ifndef LDAP_SASL_BIND_RESULT_H
#define LDAP_SASL_BIND_RESULT_H

#include <ldap.h>

#include <LDAPResult.h>

class LDAPRequest;

/**
 * Object of this class are created by the LDAPMsg::create method if
 * results for an Extended Operation were returned by a LDAP server.
 */
class LDAPSaslBindResult : public LDAPResult {
    public :
        /**
         * Constructor that creates an LDAPExtResult-object from the C-API
         * structures
         */
        LDAPSaslBindResult(const LDAPRequest* req, LDAPMessage* msg);

        /**
         * The Destructor
         */
        virtual ~LDAPSaslBindResult();

        /**
         * @returns If the result contained data this method will return
         *          the data to the caller as a std::string.
         */
        const std::string& getServerCreds() const;

    private:
        std::string m_creds;
};

#endif // LDAP_SASL_BIND_RESULT_H
