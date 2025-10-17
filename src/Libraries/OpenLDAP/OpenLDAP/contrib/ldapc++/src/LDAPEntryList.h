/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 14, 2024.
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

#ifndef LDAP_ENTRY_LIST_H
#define LDAP_ENTRY_LIST_H

#include <cstdio>
#include <list>

class LDAPEntry;
   
/**
 * For internal use only.
 * 
 * This class is used by LDAPSearchResults to store a std::list of
 * LDAPEntry-Objects
 */
class LDAPEntryList{
    typedef std::list<LDAPEntry> ListType;

    public:
	typedef ListType::const_iterator const_iterator;

        /**
         * Copy-Constructor
         */
        LDAPEntryList(const LDAPEntryList& el);

        /**
         * Default-Constructor
         */
        LDAPEntryList();

        /**
         * Destructor
         */
        ~LDAPEntryList();

        /**
         * @return The number of entries currently stored in the list.
         */
        size_t size() const;

        /**
         * @return true if there are zero entries currently stored in the list.
         */
        bool empty() const;

        /**
         * @return An iterator pointing to the first element of the list.
         */
        const_iterator begin() const;

        /**
         * @return An iterator pointing to the end of the list
         */
        const_iterator end() const;

        /**
         * Adds an Entry to the end of the list.
         */
        void addEntry(const LDAPEntry& e);

    private:
        ListType m_entries;
};
#endif // LDAP_ENTRY_LIST_H
