/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 11, 2024.
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


#include <ldap.h>

#include "debug.h"

#include "LDAPAddRequest.h"
#include "LDAPEntry.h"
#include "LDAPException.h"
#include "LDAPMessageQueue.h"
#include "LDAPResult.h"

using namespace std;

LDAPAddRequest::LDAPAddRequest(const LDAPAddRequest& req) :
        LDAPRequest(req){
    DEBUG(LDAP_DEBUG_CONSTRUCT, "LDAPAddRequest::LDAPAddRequest(&)" << endl);
    m_entry=new LDAPEntry(*(req.m_entry));
}

LDAPAddRequest::LDAPAddRequest(const LDAPEntry* entry, 
        LDAPAsynConnection *connect, const LDAPConstraints *cons,
        bool isReferral, const LDAPRequest* parent) 
        : LDAPRequest(connect, cons, isReferral,parent){
    DEBUG(LDAP_DEBUG_CONSTRUCT, "LDAPAddRequest::LDAPAddRequest()" << endl);
    DEBUG(LDAP_DEBUG_CONSTRUCT | LDAP_DEBUG_PARAMETER, 
            "   entry:" << entry << endl 
            << "   isReferral:" << isReferral << endl);
    m_requestType = LDAPRequest::ADD;
    m_entry = new LDAPEntry(*entry);
}

LDAPAddRequest::~LDAPAddRequest(){
    DEBUG(LDAP_DEBUG_DESTROY, "LDAPAddRequest::~LDAPAddRequest()" << endl);
    delete m_entry;
}

LDAPMessageQueue* LDAPAddRequest::sendRequest(){
    DEBUG(LDAP_DEBUG_TRACE, "LDAPAddRequest::sendRequest()" << endl);
    int msgID=0;
    const LDAPAttributeList* list=m_entry->getAttributes();
    LDAPMod** attrs=list->toLDAPModArray();
    LDAPControl** tmpSrvCtrls = m_cons->getSrvCtrlsArray();
    LDAPControl** tmpClCtrls = m_cons->getClCtrlsArray();
    int err=ldap_add_ext(m_connection->getSessionHandle(),
            m_entry->getDN().c_str(),attrs,tmpSrvCtrls,tmpClCtrls,&msgID);
    LDAPControlSet::freeLDAPControlArray(tmpSrvCtrls);
    LDAPControlSet::freeLDAPControlArray(tmpClCtrls);
    ldap_mods_free(attrs,1);
    if(err != LDAP_SUCCESS){
        throw LDAPException(err);
    }else{
        m_msgID=msgID;
        return new LDAPMessageQueue(this);
    }
}

LDAPRequest* LDAPAddRequest::followReferral(LDAPMsg* ref){
    DEBUG(LDAP_DEBUG_TRACE, "LDAPAddRequest::followReferral()"<< endl);
    LDAPUrlList::const_iterator usedUrl;
    LDAPUrlList urls = ((LDAPResult*)ref)->getReferralUrls();
    LDAPAsynConnection* con = 0;
    try {
        con = getConnection()->referralConnect(urls,usedUrl,m_cons);
    } catch(LDAPException e){
        delete con;
        return 0;
    }
    if(con != 0){
        return new LDAPAddRequest(m_entry, con, m_cons,true,this);
    }
    return 0;
}

