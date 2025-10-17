/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 14, 2023.
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
#include <ldap.h>
#include <lber.h>

#include "debug.h"

#include "LDAPExtRequest.h"
#include "LDAPException.h"
#include "LDAPResult.h"

#include <cstdlib>

using namespace std;

LDAPExtRequest::LDAPExtRequest(const LDAPExtRequest& req) :
        LDAPRequest(req){
    DEBUG(LDAP_DEBUG_CONSTRUCT,"LDAPExtRequest::LDAPExtRequest(&)" << endl);
    m_data=req.m_data;
    m_oid=req.m_oid;
}

LDAPExtRequest::LDAPExtRequest(const string& oid, const string& data, 
        LDAPAsynConnection *connect, const LDAPConstraints *cons,
        bool isReferral, const LDAPRequest* parent) 
        : LDAPRequest(connect, cons, isReferral, parent){
    DEBUG(LDAP_DEBUG_CONSTRUCT, "LDAPExtRequest::LDAPExtRequest()" << endl);
    DEBUG(LDAP_DEBUG_CONSTRUCT | LDAP_DEBUG_PARAMETER, 
            "   oid:" << oid << endl);
    m_oid=oid;
    m_data=data;
}

LDAPExtRequest::~LDAPExtRequest(){
    DEBUG(LDAP_DEBUG_DESTROY, "LDAPExtRequest::~LDAPExtRequest()" << endl);
}

LDAPMessageQueue* LDAPExtRequest::sendRequest(){
    DEBUG(LDAP_DEBUG_TRACE, "LDAPExtRequest::sendRequest()" << endl);
    int msgID=0;
    BerValue* tmpdata=0;
    if(m_data != ""){
        tmpdata=(BerValue*) malloc(sizeof(BerValue));
        tmpdata->bv_len = m_data.size();
        tmpdata->bv_val = (char*) malloc(sizeof(char) * (m_data.size()) );
        m_data.copy(tmpdata->bv_val, string::npos);
    }
    LDAPControl** tmpSrvCtrls=m_cons->getSrvCtrlsArray();
    LDAPControl** tmpClCtrls=m_cons->getClCtrlsArray();
    int err=ldap_extended_operation(m_connection->getSessionHandle(),
            m_oid.c_str(), tmpdata, tmpSrvCtrls, tmpClCtrls, &msgID);
    LDAPControlSet::freeLDAPControlArray(tmpSrvCtrls);
    LDAPControlSet::freeLDAPControlArray(tmpClCtrls);
    ber_bvfree(tmpdata);
    if(err != LDAP_SUCCESS){
        delete this;
        throw LDAPException(err);
    }else{
        m_msgID=msgID;
        return new LDAPMessageQueue(this);
    }
}

LDAPRequest* LDAPExtRequest::followReferral(LDAPMsg* ref){
    DEBUG(LDAP_DEBUG_TRACE, "LDAPExtRequest::followReferral()" << endl);
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
        return new LDAPExtRequest(m_oid, m_data, con, m_cons,true,this);
    }
    return 0;
}


