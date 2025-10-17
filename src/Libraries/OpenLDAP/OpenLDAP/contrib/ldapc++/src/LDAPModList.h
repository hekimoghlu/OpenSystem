/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 11, 2024.
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


#ifndef LDAP_MOD_LIST_H
#define LDAP_MOD_LIST_H

#include <ldap.h>
#include <list>
#include <LDAPModification.h>

/**
 * This container class is used to store multiple LDAPModification-objects.
 */
class LDAPModList{
    typedef std::list<LDAPModification> ListType;

    public : 
        /**
         * Constructs an empty list.
         */   
        LDAPModList();
		
        /**
         * Copy-constructor
         */
        LDAPModList(const LDAPModList&);

        /**
         * Adds one element to the end of the list.
         * @param mod The LDAPModification to add to the std::list.
         */
        void addModification(const LDAPModification &mod);

        /**
         * Translates the list to a 0-terminated array of
         * LDAPMod-structures as needed by the C-API
         */
        LDAPMod** toLDAPModArray();

        /**
         * @returns true, if the ModList contains no Operations
         */
        bool empty() const;
        
        /**
         * @returns number of Modifications in the ModList
         */
        unsigned int size() const;

    private : 
        ListType m_modList;
};
#endif //LDAP_MOD_LIST_H


