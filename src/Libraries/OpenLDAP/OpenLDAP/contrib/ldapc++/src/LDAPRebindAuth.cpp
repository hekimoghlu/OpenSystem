/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 16, 2024.
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

#include <iostream>

#include "LDAPRebindAuth.h"
#include "debug.h"

using namespace std;

LDAPRebindAuth::LDAPRebindAuth(const string& dn, const string& pwd){
    DEBUG(LDAP_DEBUG_CONSTRUCT,"LDAPRebindAuth::LDAPRebindAuth()" << endl);
    DEBUG(LDAP_DEBUG_CONSTRUCT | LDAP_DEBUG_PARAMETER,"   dn:" << dn << endl 
            << "   pwd:" << pwd << endl);
    m_dn=dn;
    m_password=pwd;
}

LDAPRebindAuth::LDAPRebindAuth(const LDAPRebindAuth& lra){
    DEBUG(LDAP_DEBUG_CONSTRUCT,"LDAPRebindAuth::LDAPRebindAuth(&)" << endl);
    m_dn=lra.m_dn;
    m_password=lra.m_password;
}

LDAPRebindAuth::~LDAPRebindAuth(){
    DEBUG(LDAP_DEBUG_DESTROY,"LDAPRebindAuth::~LDAPRebindAuth()" << endl);
}

const string& LDAPRebindAuth::getDN() const{
    DEBUG(LDAP_DEBUG_TRACE,"LDAPRebindAuth::getDN()" << endl);
    return m_dn;
}

const string& LDAPRebindAuth::getPassword() const{
    DEBUG(LDAP_DEBUG_TRACE,"LDAPRebindAuth::getPassword()" << endl);
    return m_password;
}
