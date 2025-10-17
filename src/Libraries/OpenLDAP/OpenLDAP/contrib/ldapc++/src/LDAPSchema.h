/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 29, 2024.
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

#ifndef LDAP_SCHEMA_H
#define LDAP_SCHEMA_H

#include <string>
#include <map>

#include "LDAPObjClass.h"
#include "LDAPAttrType.h"

/**
 * Represents the LDAP schema
 */
class LDAPSchema{
    private :
	/**
	 * map of object classes: index is name, value is LDAPObjClass object
	 */
	map <string, LDAPObjClass> object_classes;
	
	/**
	 * map of attribute types: index is name, value is LDAPAttrType object
	 */
	map <string, LDAPAttrType> attr_types;

    public :

        /**
         * Constructs an empty object
         */   
        LDAPSchema();

        /**
         * Destructor
         */
        virtual ~LDAPSchema();
	
        /**
         * Fill the object_classes map
	 * @param oc description of one objectclass (string returned by search
	 * command), in form:
	 * "( 1.2.3.4.5 NAME '<name>' SUP <supname> STRUCTURAL
	 *    DESC '<description>' MUST ( <attrtype> ) MAY ( <attrtype> ))"
         */
	void setObjectClasses (const StringList &oc);

	 /**
         * Fill the attr_types map
	 * @param at description of one attribute type
	 *  (string returned by search command), in form:
	 * "( 1.2.3.4.6 NAME ( '<name>' ) DESC '<desc>'
	 *    EQUALITY caseExactIA5Match SYNTAX 1.3.6.1.4.1.1466.115.121.1.26 )"
         */
	void setAttributeTypes (const StringList &at);

	/**
	 * Returns object class object with given name
	 */
	LDAPObjClass getObjectClassByName (std::string name);
	
	/**
	 * Returns attribute type object with given name
	 */
	LDAPAttrType getAttributeTypeByName (string name);

};

#endif // LDAP_SCHEMA_H
