/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 15, 2025.
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

#include "LDAPModifyRequest.h"
#include "LDAPException.h"
#include "LDAPMessageQueue.h"
#include "LDAPResult.h"

using namespace std;

LDAPModifyRequest::LDAPModifyRequest(const LDAPModifyRequest& req) :
        LDAPRequest(req){
    DEBUG(LDAP_DEBUG_CONSTRUCT, 
            "LDAPModifyRequest::LDAPModifyRequest(&)" << endl);
    m_modList = new LDAPModList(*(req.m_modList));
    m_dn = req.m_dn;
}

LDAPModifyRequest::LDAPModifyRequest(const string& dn, 
        const LDAPModList *modList, LDAPAsynConnection *connect,
        const LDAPConstraints *cons, bool isReferral,
        const LDAPRequest* parent) :
        LDAPRequest(connect, cons, isReferral, parent){
    DEBUG(LDAP_DEBUG_CONSTRUCT, 
            "LDAPModifyRequest::LDAPModifyRequest(&)" << endl);            
    DEBUG(LDAP_DEBUG_CONSTRUCT | LDAP_DEBUG_PARAMETER, 
            "   dn:" << dn << endl);
    m_dn = dn;
    m_modList = new LDAPModList(*modList);
}

LDAPModifyRequest::~LDAPModifyRequest(){
    DEBUG(LDAP_DEBUG_DESTROY, 
            "LDAPModifyRequest::~LDAPModifyRequest()" << endl);
    delete m_modList;
}

LDAPMessageQueue* LDAPModifyRequest::sendRequest(){
    DEBUG(LDAP_DEBUG_TRACE, "LDAPModifyRequest::sendRequest()" << endl);
    int msgID=0;
    LDAPControl** tmpSrvCtrls=m_cons->getSrvCtrlsArray();
    LDAPControl** tmpClCtrls=m_cons->getClCtrlsArray();
    LDAPMod** tmpMods=m_modList->toLDAPModArray();
    int err=ldap_modify_ext(m_connection->getSessionHandle(),m_dn.c_str(),
            tmpMods, tmpSrvCtrls, tmpClCtrls,&msgID);
    LDAPControlSet::freeLDAPControlArray(tmpSrvCtrls);
    LDAPControlSet::freeLDAPControlArray(tmpClCtrls);
    ldap_mods_free(tmpMods,1);
    if(err != LDAP_SUCCESS){
        throw LDAPException(err);
    }else{
        m_msgID=msgID;
        return new LDAPMessageQueue(this);
    }
}

LDAPRequest* LDAPModifyRequest::followReferral(LDAPMsg* ref){
    DEBUG(LDAP_DEBUG_TRACE, "LDAPModifyRequest::followReferral()" << endl);
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
        return new LDAPModifyRequest(m_dn, m_modList, con, m_cons,true,this);
    }
    return 0;
}


