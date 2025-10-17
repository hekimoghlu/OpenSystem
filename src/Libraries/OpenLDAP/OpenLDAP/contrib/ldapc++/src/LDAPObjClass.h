/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 18, 2022.
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

#ifndef LDAP_OBJCLASS_H
#define LDAP_OBJCLASS_H

#include <ldap_schema.h>
#include <string>

#include "StringList.h"

using namespace std;

/**
 * Represents the Object Class (from LDAP schema)
 */
class LDAPObjClass{
    private :
	StringList names, must, may, sup;
	string desc, oid;
	int kind;
	
    public :

        /**
         * Constructs an empty object.
         */   
        LDAPObjClass();

        /**
         * Copy constructor
	 */   
	LDAPObjClass( const LDAPObjClass& oc );

        /**
	 * Constructs new object and fills the data structure by parsing the
	 * argument.
	 * @param oc_item description of object class is string returned
	 * by the search command. It is in the form:
	 * "( SuSE.YaST.OC:5 NAME 'userTemplate' SUP objectTemplate STRUCTURAL
	 *    DESC 'User object template' MUST ( cn ) MAY ( secondaryGroup ))"
         */   
        LDAPObjClass (string oc_item, int flags = LDAP_SCHEMA_ALLOW_NO_OID |
                      LDAP_SCHEMA_ALLOW_QUOTED);

        /**
         * Destructor
         */
        virtual ~LDAPObjClass();
	
	/**
	 * Returns object class description
	 */
	string getDesc() const;
	
	/**
	 * Returns object class oid
	 */
	string getOid() const;

	/**
	 * Returns object class name (first one if there are more of them)
	 */
	string getName() const;

	/**
	 * Returns object class kind: 0=ABSTRACT, 1=STRUCTURAL, 2=AUXILIARY
	 */
	int getKind() const;

	/**
	 * Returns all object class names
	 */
	StringList getNames() const;
	
	/**
	 * Returns list of required attributes
	 */
	StringList getMust() const;
	
	/**
	 * Returns list of allowed (and not required) attributes
	 */
	StringList getMay() const;
	
        /**
	 * Returns list of the OIDs of the superior ObjectClasses
	 */
	StringList getSup() const;

	void setNames (char **oc_names);
	void setMay (char **oc_may);
	void setMust (char **oc_must);
	void setDesc (char *oc_desc);
	void setOid (char *oc_oid);
	void setKind (int oc_kind);
	void setSup (char **oc_sup);
	
};

#endif // LDAP_OBJCLASS_H
