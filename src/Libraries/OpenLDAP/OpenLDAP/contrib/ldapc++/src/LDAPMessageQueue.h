/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 13, 2023.
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


#ifndef LDAP_MESSAGE_QUEUE_H
#define LDAP_MESSAGE_QUEUE_H

#include <stack>

#include <LDAPUrlList.h>
#include <LDAPMessage.h>

class LDAPAsynConnection;
class LDAPRequest;
class LDAPSearchRequest;
class LDAPUrl;
typedef std::stack<LDAPRequest*> LDAPRequestStack;
typedef std::list<LDAPRequest*> LDAPRequestList;

/**
 * This class is created for the asynchronous LDAP-operations. And can be
 * used by the client to retrieve the results of an operation.
 */
class LDAPMessageQueue{
    public :

        /**
         * This creates a new LDAPMessageQueue. For a LDAP-request
         *
         * @param conn  The Request for that is queue can be used to get
         *              the results.
         */
        LDAPMessageQueue(LDAPRequest *conn);
        /**
         * Destructor
         */
        ~LDAPMessageQueue();

        /**
         * This method reads exactly one Message from the results of a
         * Request. 
         * @throws LDAPException
         * @return A pointer to an object of one of the classes that were
         *          derived from LDAPMsg. The user has to cast it to the
         *          correct type (e.g. LDAPResult or LDAPSearchResult)
         */           
        LDAPMsg* getNext();

        /**
         * For internat use only.
         *
         * The method is used to start the automatic referral chasing
         */
        LDAPRequest* chaseReferral(LDAPMsg* ref);

        /**
         * For internal use only
         *
         * The referral chasing algorithm needs this method to see the
         * currently active requests.
         */
        LDAPRequestStack* getRequestStack(); 
    
    private :
        LDAPRequestStack m_activeReq;
        LDAPRequestList m_issuedReq;
};
#endif //ifndef LDAP_MESSAGE_QUEUE_H

