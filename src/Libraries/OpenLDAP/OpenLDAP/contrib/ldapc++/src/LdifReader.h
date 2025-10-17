/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 31, 2024.
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
 * Copyright 2008-2011 The OpenLDAP Foundation, All Rights Reserved.
 * COPYING RESTRICTIONS APPLY, see COPYRIGHT file
 */

#ifndef LDIF_READER_H
#define LDIF_READER_H

#include <LDAPEntry.h>
#include <iosfwd>
#include <list>

typedef std::list< std::pair<std::string, std::string> > LdifRecord;
class LdifReader
{
    public:
        LdifReader( std::istream &input );

        inline bool isEntryRecords() const
        {
            return !m_ldifTypeRequest;
        }

        inline bool isChangeRecords() const
        {
            return m_ldifTypeRequest;
        }

        inline int getVersion() const
        {
            return m_version;
        }

        LDAPEntry getEntryRecord();
        int readNextRecord( bool first=false );
        //LDAPRequest getChangeRecord();

    private:
        int getLdifLine(std::string &line);

        void splitLine(const std::string& line, 
                    std::string &type,
                    std::string &value ) const;

        std::string readIncludeLine( const std::string &line) const;

        std::istream &m_ldifstream;
        LdifRecord m_currentRecord;
        int m_version;
        int m_curRecType;
        int m_lineNumber;
        bool m_ldifTypeRequest;
        bool m_currentIsFirst;
};

#endif /* LDIF_READER_H */
