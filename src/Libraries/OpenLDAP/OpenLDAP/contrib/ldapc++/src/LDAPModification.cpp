/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 24, 2022.
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


#include "LDAPModification.h"
#include "debug.h"

using namespace std;

LDAPModification::LDAPModification(const LDAPAttribute& attr, mod_op op){
    DEBUG(LDAP_DEBUG_CONSTRUCT,"LDAPModification::LDAPModification()" << endl);
    DEBUG(LDAP_DEBUG_CONSTRUCT | LDAP_DEBUG_PARAMETER,
            "   attr:" << attr << endl);
    m_attr = attr;
    m_mod_op = op;
}

LDAPMod* LDAPModification::toLDAPMod() const  {
    DEBUG(LDAP_DEBUG_TRACE,"LDAPModification::toLDAPMod()" << endl);
    LDAPMod* ret=m_attr.toLDAPMod();

    //The mod_op value of the LDAPMod-struct needs to be ORed with the right
    // LDAP_MOD_* constant to preserve the BIN-flag (see CAPI-draft for 
    // explanation of the LDAPMod struct)
    switch (m_mod_op){
	case OP_ADD :
	    ret->mod_op |= LDAP_MOD_ADD;
	break;
	case OP_DELETE :
	    ret->mod_op |= LDAP_MOD_DELETE;
	break;
	case OP_REPLACE :
	    ret->mod_op |= LDAP_MOD_REPLACE;
	break;
    }
    return ret;
}

const LDAPAttribute* LDAPModification::getAttribute() const {
	return &m_attr;
}

LDAPModification::mod_op LDAPModification::getOperation() const {
	return m_mod_op;
}
