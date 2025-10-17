/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 14, 2025.
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


#ifndef LDAP_ENTRY_H
#define LDAP_ENTRY_H
#include <ldap.h>

#include <LDAPAttributeList.h>

class LDAPAsynConnection;

/**
 * This class is used to store every kind of LDAP Entry.
 */
class LDAPEntry{

    public :
        /**
         * Copy-constructor
         */
        LDAPEntry(const LDAPEntry& entry);

        /**
         * Constructs a new entry (also used as standard constructor).
         *
         * @param dn    The Distinguished Name for the new entry.
         * @param attrs The attributes for the new entry.
         */
        LDAPEntry(const std::string& dn=std::string(), 
                const LDAPAttributeList *attrs=0);

        /**
         * Used internally only.
         *
         * The constructor is used internally to create a LDAPEntry from
         * the C-API's data structurs.
         */ 
        LDAPEntry(const LDAPAsynConnection *ld, LDAPMessage *msg);

        /**
         * Destructor
         */
        ~LDAPEntry();

        /**
         * Assignment operator
         */
        LDAPEntry& operator=(const LDAPEntry& from);

        /**
         * Sets the DN-attribute.
         * @param dn: The new DN for the entry.
         */
        void setDN(const std::string& dn);

        /**
         * Sets the attributes of the entry.
         * @param attr: A pointer to a std::list of the new attributes.
         */
        void setAttributes(LDAPAttributeList *attrs);

	/**
	 * Get an Attribute by its AttributeType (simple wrapper around
         * LDAPAttributeList::getAttributeByName() )
	 * @param name The name of the Attribute to look for
	 * @return a pointer to the LDAPAttribute with the AttributeType 
	 *	"name" or 0, if there is no Attribute of that Type
	 */
	const LDAPAttribute* getAttributeByName(const std::string& name) const;

        /**
         * Adds one Attribute to the List of Attributes (simple wrapper around
         * LDAPAttributeList::addAttribute() ).
         * @param attr The attribute to add to the list.
         */
        void addAttribute(const LDAPAttribute& attr);
        
        /**
         * Deletes all values of an Attribute from the list of Attributes 
         * (simple wrapper around LDAPAttributeList::delAttribute() ).
         * @param type The attribute to delete.
         */
        void delAttribute(const std::string& type);

        /**
         * Replace an Attribute in the List of Attributes (simple wrapper
         * around LDAPAttributeList::replaceAttribute() ).
         * @param attr The attribute to add to the list.
         */
        void replaceAttribute(const LDAPAttribute& attr);

        /**
         * @returns The current DN of the entry.
         */
        const std::string& getDN() const ;

        /**
         * @returns A const pointer to the attributes of the entry.  
         */
        const LDAPAttributeList* getAttributes() const;

        /**
         * This method can be used to dump the data of a LDAPResult-Object.
         * It is only useful for debugging purposes at the moment
         */
        friend std::ostream& operator << (std::ostream& s, const LDAPEntry& le);
	
    private :
        LDAPAttributeList *m_attrs;
        std::string m_dn;
};
#endif  //LDAP_ENTRY_H
