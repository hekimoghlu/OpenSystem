/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 8, 2025.
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
/*
    DLDbList.h
    
    This implements a vector of DLDbIdentifiers. A DLDbIdentifier contains all of the
    information needed to find a particular DB within a particular DL. This file
    does not depend on CoreFoundation but does depend on CDSA headers.
*/

#ifndef _H_CDSA_CLIENT_DLDBLIST
#define _H_CDSA_CLIENT_DLDBLIST  1

#include <security_cdsa_utilities/cssmdb.h>
#include <security_utilities/refcount.h>
#include <vector>

namespace Security
{

namespace CssmClient
{

//-------------------------------------------------------------------------------------
//
//			Lists of DL/DBs
//
//-------------------------------------------------------------------------------------


//
// DLDbList
//
class DLDbList : public vector<DLDbIdentifier>
{
public:
    DLDbList() : mChanged(false) {}
    virtual ~DLDbList() {}
    
    // API
    virtual void add(const DLDbIdentifier& dldbIdentifier);		// Adds at end if not in list
    virtual void remove(const DLDbIdentifier& dldbIdentifier);	// Removes from list
    virtual void save();

    bool hasChanged() const { return mChanged; }

protected:
    void changed(bool hasChanged) { mChanged=hasChanged; }
    
private:
    bool mChanged;
};

}; // end namespace CssmClient

} // end namespace Security

#endif // _H_CDSA_CLIENT_DLDBLIST
