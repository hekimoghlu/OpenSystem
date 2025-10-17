/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 27, 2024.
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

#ifndef LDAP_REFERENCE_LIST_H
#define LDAP_REFERENCE_LIST_H

#include <cstdio>
#include <list>

class LDAPSearchReference;

/**
 * Container class for storing a list of Search References
 *
 * Used internally only by LDAPSearchResults
 */
class LDAPReferenceList{
    typedef std::list<LDAPSearchReference> ListType;

    public:
	typedef ListType::const_iterator const_iterator;

        /**
         * Constructs an empty list.
         */   
        LDAPReferenceList();

        /**
         * Copy-constructor
         */
        LDAPReferenceList(const LDAPReferenceList& rl);

        /**
         * Destructor
         */
        ~LDAPReferenceList();

        /**
         * @return The number of LDAPSearchReference-objects that are 
         * currently stored in this list.
         */
        size_t size() const;

        /**
         * @return true if there are zero LDAPSearchReference-objects
         * currently stored in this list.
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
         * Adds one element to the end of the list.
         * @param e The LDAPSearchReference to add to the list.
         */
        void addReference(const LDAPSearchReference& e);

    private:
        ListType m_refs;
};
#endif // LDAP_REFERENCE_LIST_H

