/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 20, 2025.
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


#include "debug.h"
#include"LDAPResult.h"
#include"LDAPAsynConnection.h"
#include "LDAPRequest.h"
#include "LDAPException.h"

#include <cstdlib>

using namespace std;

LDAPResult::LDAPResult(const LDAPRequest *req, LDAPMessage *msg) : 
        LDAPMsg(msg){
    if(msg != 0){
        DEBUG(LDAP_DEBUG_CONSTRUCT,"LDAPResult::LDAPResult()" << endl);
        const LDAPAsynConnection *con=req->getConnection();
        char **refs=0;
        LDAPControl** srvctrls=0;
        char* matchedDN=0;
        char* errMsg=0;
        int err=ldap_parse_result(con->getSessionHandle(),msg,&m_resCode,
                &matchedDN, &errMsg,&refs,&srvctrls,0);
        if(err != LDAP_SUCCESS){
            ber_memvfree((void**) refs);
            ldap_controls_free(srvctrls);
            throw LDAPException(err);
        }else{
            if (refs){
                m_referrals=LDAPUrlList(refs);
                ber_memvfree((void**) refs);
            }
            if (srvctrls){
                m_srvControls = LDAPControlSet(srvctrls);
                m_hasControls = true;
                ldap_controls_free(srvctrls);
            }else{
                m_hasControls = false;
            }
            if(matchedDN != 0){
                m_matchedDN=string(matchedDN);
                free(matchedDN);
            }
            if(errMsg != 0){
                m_errMsg=string(errMsg);
                free(errMsg);
            }
        }
    }
}

LDAPResult::LDAPResult(int type, int resultCode, const std::string &msg) : 
        LDAPMsg(type,0), m_resCode(resultCode), m_errMsg(msg)
{}


LDAPResult::~LDAPResult(){
    DEBUG(LDAP_DEBUG_DESTROY,"LDAPResult::~LDAPResult()" << endl);
}

int LDAPResult::getResultCode() const{
    DEBUG(LDAP_DEBUG_TRACE,"LDAPResult::getResultCode()" << endl);
    return m_resCode;
}

string LDAPResult::resToString() const{
    DEBUG(LDAP_DEBUG_TRACE,"LDAPResult::resToString()" << endl);
    return string(ldap_err2string(m_resCode));
}

const string& LDAPResult::getErrMsg() const{
    DEBUG(LDAP_DEBUG_TRACE,"LDAPResult::getErrMsg()" << endl);
    return m_errMsg;
}

const string& LDAPResult::getMatchedDN() const{
    DEBUG(LDAP_DEBUG_TRACE,"LDAPResult::getMatchedDN()" << endl);
    return m_matchedDN;
}

const LDAPUrlList& LDAPResult::getReferralUrls() const{
    DEBUG(LDAP_DEBUG_TRACE,"LDAPResult::getReferralUrl()" << endl);
    return m_referrals;
}

ostream& operator<<(ostream &s,LDAPResult &l){
    return s << "Result: " << l.m_resCode << ": "  
        << ldap_err2string(l.m_resCode) << endl 
        << "Matched: " << l.m_matchedDN << endl << "ErrMsg: " << l.m_errMsg;
}

