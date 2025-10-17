/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 3, 2024.
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

#ifndef STRING_LIST_H
#define STRING_LIST_H

#include <string>
#include <list>
/**
 * Container class to store multiple string-objects
 */
class StringList{
    typedef std::list<std::string> ListType;

    private:
        ListType m_data;

    public:
	typedef ListType::const_iterator const_iterator;
   
        /**
         * Constructs an empty list.
         */   
        StringList();

        /**
         * Copy-constructor
         */
        StringList(const StringList& sl);

        /**
         * For internal use only
         *
         * This constructor is used by the library internally to create a
         * list of string from a array for c-Strings (char*)thar was
         * returned by the C-API
         */
        StringList(char** values);

        /**
         * Destructor
         */
        ~StringList();
    
        /**
         * The methods converts the list to a 0-terminated array of
         * c-Strings.
         */
        char** toCharArray() const;
  
        /**
         * Adds one element to the end of the list.
         * @param attr The attribute to add to the list.
         */
        void add(const std::string& value);

        /**
         * @return The number of strings that are currently
         * stored in this list.
         */
        size_t size() const;

        /**
         * @return true if there are zero strings currently
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
         * removes all elements from the list
         */
        void clear(); 
};
#endif //STRING_LIST_H
