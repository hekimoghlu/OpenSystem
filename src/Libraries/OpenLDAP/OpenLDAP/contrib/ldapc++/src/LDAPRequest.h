/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 6, 2021.
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


#ifndef LDAP_REQUEST_H
#define LDAP_REQUEST_H

#include <LDAPConstraints.h>
#include <LDAPAsynConnection.h>
#include <LDAPMessageQueue.h>

class LDAPUrl;

/**
 * For internal use only
 * 
 * Each request that is sent to a LDAP-server by this library is
 * represented by a special object that contains the parameters and some
 * other info of the request. This virtual class is the common base classe
 * for these specialized request classes.
 */
class LDAPRequest{

    public :
        static const int BIND=0;
        static const int UNBIND=2;
        static const int SEARCH=3;
        static const int MODIFY=7;
        static const int ADD=8;
		static const int DELETE=10;
        static const int COMPARE=14;

        LDAPRequest(const LDAPRequest& req);
        LDAPRequest(LDAPAsynConnection* conn, 
                const LDAPConstraints* cons, bool isReferral=false,
                const LDAPRequest* parent=0);
        virtual ~LDAPRequest();
        
        const LDAPConstraints* getConstraints() const;
        const LDAPAsynConnection* getConnection() const;
        virtual LDAPMsg *getNextMessage() const;
        int getType()const;
        int getMsgID() const;
        int getHopCount() const;

        /**
         * @return The LDAPRequest that has created this object. Or 0 if
         * this object was not created by another request.
         */
        const LDAPRequest* getParent() const;

        /**
         * @return true if this object was created during the automatic
         * chasing of referrals. Otherwise false
         */
        bool isReferral() const;
        
        void unbind() const; 

        /**
         * This method encodes the request an calls the apprpriate
         * functions of the C-API to send the Request to a LDAP-Server
         */
        virtual LDAPMessageQueue* sendRequest()=0;
        virtual LDAPRequest* followReferral(LDAPMsg* ref);

        /**
         * Compare this request with another on. And returns true if they
         * have the same parameters.
         */
        virtual bool equals(const LDAPRequest* req) const;

        bool isCycle() const;
        
    protected :
        bool m_isReferral;
        int m_requestType;
        LDAPConstraints *m_cons;
        LDAPAsynConnection *m_connection;
        const LDAPRequest* m_parent;
        int m_hopCount;
        int m_msgID;  //the associated C-API Message ID
        LDAPRequest();
};
#endif //LDAP_REQUEST_H 

