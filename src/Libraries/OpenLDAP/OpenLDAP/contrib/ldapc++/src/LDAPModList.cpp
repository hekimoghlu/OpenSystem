/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 1, 2022.
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


#include "LDAPModList.h"
#include "debug.h"

#include <cstdlib>

using namespace std;

LDAPModList::LDAPModList(){
    DEBUG(LDAP_DEBUG_CONSTRUCT,"LDAPModList::LDAPModList()" << endl);
}

LDAPModList::LDAPModList(const LDAPModList& ml){
    DEBUG(LDAP_DEBUG_CONSTRUCT,"LDAPModList::LDAPModList(&)" << endl);
    m_modList=ml.m_modList;
}

void LDAPModList::addModification(const LDAPModification &mod){
    DEBUG(LDAP_DEBUG_TRACE,"LDAPModList::addModification()" << endl);
	m_modList.push_back(mod);
}

LDAPMod** LDAPModList::toLDAPModArray(){
    DEBUG(LDAP_DEBUG_TRACE,"LDAPModList::toLDAPModArray()" << endl);
    LDAPMod **ret = (LDAPMod**) malloc(
		    (m_modList.size()+1) * sizeof(LDAPMod*));
    ret[m_modList.size()]=0;
    LDAPModList::ListType::const_iterator i;
    int j=0;
    for (i=m_modList.begin(); i != m_modList.end(); i++ , j++){
	    ret[j]=i->toLDAPMod();
    }
    return ret;
}

bool LDAPModList::empty() const {
    return m_modList.empty();
}

unsigned int LDAPModList::size() const {
    return m_modList.size();
}
