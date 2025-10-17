/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 30, 2023.
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
 * Copyright 2000-2011 The OpenLDAP Foundation, All Rights Reserved.
 * COPYING RESTRICTIONS APPLY, see COPYRIGHT file
 */

#ifndef LDAP_BIND_REQUEST_H
#define LDAP_BIND_REQUEST_H

#include <LDAPRequest.h>
#include <LDAPResult.h>
#include <SaslInteractionHandler.h>

class LDAPBindRequest : LDAPRequest {
    private:
        std::string m_dn;
        std::string m_cred;
        std::string m_mech;

    public:
        LDAPBindRequest( const LDAPBindRequest& req);
        //just for simple authentication
        LDAPBindRequest(const std::string&, const std::string& passwd, 
                LDAPAsynConnection *connect, const LDAPConstraints *cons, 
                bool isReferral=false);
        virtual ~LDAPBindRequest();
        virtual LDAPMessageQueue *sendRequest();
};

class LDAPSaslBindRequest : LDAPRequest
{
    public:
        LDAPSaslBindRequest( const std::string& mech, const std::string& cred, 
        LDAPAsynConnection *connect, const LDAPConstraints *cons, 
                bool isReferral=false);
        virtual LDAPMessageQueue *sendRequest();
        virtual ~LDAPSaslBindRequest();

    private:
        std::string m_mech;
        std::string m_cred;
};

class LDAPSaslInteractiveBind : LDAPRequest
{
    public:
        LDAPSaslInteractiveBind( const std::string& mech, int flags,
                SaslInteractionHandler *sih, LDAPAsynConnection *connect, 
                const LDAPConstraints *cons, bool isReferral=false);
        virtual LDAPMessageQueue *sendRequest();
        virtual LDAPMsg* getNextMessage() const;
        virtual ~LDAPSaslInteractiveBind();

    private:
        std::string m_mech;
        int m_flags;
        SaslInteractionHandler *m_sih;
        LDAPResult *m_res;
};
#endif //LDAP_BIND_REQUEST_H

