/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 23, 2024.
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
 * Copyright 2003-2011 The OpenLDAP Foundation, All Rights Reserved.
 * COPYING RESTRICTIONS APPLY, see COPYRIGHT file
 */

#include "LDAPSchema.h"

#include <ctype.h>
#include <ldap.h>

#include "debug.h"
#include "StringList.h"


using namespace std;

LDAPSchema::LDAPSchema(){
    DEBUG(LDAP_DEBUG_CONSTRUCT,
            "LDAPSchema::LDAPSchema( )" << endl);
}

LDAPSchema::~LDAPSchema() {
    DEBUG(LDAP_DEBUG_DESTROY,"LDAPSchema::~LDAPSchema()" << endl);
}

void LDAPSchema::setObjectClasses (const StringList &ocs) {
    DEBUG(LDAP_DEBUG_TRACE,"LDAPSchema::setObjectClasses()" << endl);
    
    // parse the stringlist and save it to global map...
    StringList::const_iterator i,j;
    for (i = ocs.begin(); i != ocs.end(); i++) {
	LDAPObjClass oc ( (*i) );
	StringList names = oc.getNames();
	// there could be more names for one object...
	for (j = names.begin(); j != names.end(); j++) {
            string lc_name = *j;
            string::iterator k;
            for ( k = lc_name.begin(); k != lc_name.end(); k++ ) {
                (*k) = tolower(*k); 
            }
	    object_classes [lc_name] = LDAPObjClass (oc);
	}
    }
}

void LDAPSchema::setAttributeTypes (const StringList &ats) {
    DEBUG(LDAP_DEBUG_TRACE,"LDAPSchema::setAttributeTypes()" << endl);
    
    // parse the stringlist and save it to global map...
    StringList::const_iterator i,j;
    for (i = ats.begin(); i != ats.end(); i++) {
	LDAPAttrType at ( (*i) );
	StringList names = at.getNames();
	// there could be more names for one object...
	for (j = names.begin(); j != names.end(); j++) {
            string lc_name = *j;
            string::iterator k;
            for ( k = lc_name.begin(); k != lc_name.end(); k++ ) {
                (*k) = tolower(*k); 
            }
	    attr_types [lc_name] = LDAPAttrType (at);
	}
    }
}

LDAPObjClass LDAPSchema::getObjectClassByName (string name) {
    string lc_name = name;
    string::iterator k;
    for ( k = lc_name.begin(); k != lc_name.end(); k++ ) {
        (*k) = tolower(*k); 
    }
    return object_classes [lc_name];
}

LDAPAttrType LDAPSchema::getAttributeTypeByName (string name) {
    string lc_name = name;
    string::iterator k;
    for ( k = lc_name.begin(); k != lc_name.end(); k++ ) {
        (*k) = tolower(*k); 
    }

    return attr_types [lc_name];
}
