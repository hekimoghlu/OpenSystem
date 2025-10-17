/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 5, 2022.
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


#ifndef LDAP_ATTRIBUTE_LIST_H
#define LDAP_ATTRIBUTE_LIST_H

#include <ldap.h>
#include <list>
#include <string>

class LDAPAttribute;
class LDAPAsynConnection;
class LDAPMsg;

/**
 * This container class is used to store multiple LDAPAttribute-objects.
 */
class LDAPAttributeList{
    typedef std::list<LDAPAttribute> ListType;

    private :
        ListType m_attrs;

    public :
        typedef ListType::const_iterator const_iterator;
	typedef ListType::iterator iterator;


        /**
         * Copy-constructor
         */
        LDAPAttributeList(const LDAPAttributeList& al);
        
        /**
         * For internal use only
         *
         * This constructor is used by the library internally to create a
         * list of attributes from a LDAPMessage-struct that was return by
         * the C-API
         */
        LDAPAttributeList(const LDAPAsynConnection *ld, LDAPMessage *msg);

        /**
         * Constructs an empty list.
         */   
        LDAPAttributeList();

        /**
         * Destructor
         */
        virtual ~LDAPAttributeList();

        /**
         * @return The number of LDAPAttribute-objects that are currently
         * stored in this list.
         */
        size_t size() const;

        /**
         * @return true if there are zero LDAPAttribute-objects currently
         * stored in this list.
         */
        bool empty() const;

        /**
         * @return A iterator that points to the first element of the list.
         */
        const_iterator begin() const;
        
        /**
         * @return A iterator that points to the element after the last
         * element of the list.
         */
        const_iterator end() const;

	/**
	 * Get an Attribute by its AttributeType
	 * @param name The name of the Attribute to look for
	 * @return a pointer to the LDAPAttribute with the AttributeType 
	 *	"name" or 0, if there is no Attribute of that Type
	 */
	const LDAPAttribute* getAttributeByName(const std::string& name) const;

        /**
         * Adds one element to the end of the list.
         * @param attr The attribute to add to the list.
         */
        void addAttribute(const LDAPAttribute& attr);
        
        /**
         * Deletes all values of an Attribute for the list
         * @param type The attribute type to be deleted.
         */
        void delAttribute(const std::string& type);

        /**
         * Replace an Attribute in the List
         * @param attr The attribute to add to the list.
         */
        void replaceAttribute(const LDAPAttribute& attr);

        /**
         * Translates the list of Attributes to a 0-terminated array of
         * LDAPMod-structures as needed by the C-API
         */
        LDAPMod** toLDAPModArray() const;
        
        /**
         * This method can be used to dump the data of a LDAPResult-Object.
         * It is only useful for debugging purposes at the moment
         */
        friend std::ostream& operator << (std::ostream& s, 
					  const LDAPAttributeList& al);
};

#endif // LDAP_ATTRIBUTE_LIST_H

