/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 5, 2024.
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
#ifndef _DBNAME_H_
#define _DBNAME_H_  1

#include <security_utilities/utilities.h>
#include <security_cdsa_utilities/walkers.h>
#include <Security/cssmtype.h>
#include <string>

#ifdef _CPP_DBNAME
# pragma export on
#endif

// @@@ Should not use using in headers.
using namespace std;

namespace Security
{

//----------------------------------------------------------------
//typedef struct cssm_net_address {
//    CSSM_NET_ADDRESS_TYPE AddressType;
//    CSSM_DATA Address;
//} CSSM_NET_ADDRESS, *CSSM_NET_ADDRESS_PTR;
//----------------------------------------------------------------

// XXX TODO: Make CssmNetAddress use a factory to constuct netadrress objects based on CSSM_NET_ADDRESS_TYPE!
class CssmNetAddress : public PodWrapper<CssmNetAddress, CSSM_NET_ADDRESS>
{
public:
    // Create a CssmNetAddress wrapper.  Copies inAddress.Data
    CssmNetAddress(CSSM_DB_RECORDTYPE inAddressType, const CssmData &inAddress);
    CssmNetAddress(const CSSM_NET_ADDRESS &other);
    ~CssmNetAddress();
    CSSM_DB_RECORDTYPE addressType() const { return AddressType; }
    const CssmData &address() const { return CssmData::overlay(Address); }
    bool operator <(const CssmNetAddress &other) const
    {
        return AddressType != other.AddressType ? AddressType < other.AddressType : address() < other.address();
    }
};

class DbName
{
public:
    DbName (const char *inDbName = NULL, const CSSM_NET_ADDRESS *inDbLocation = NULL);
    DbName(const DbName &other);
    DbName &operator =(const DbName &other);
    ~DbName ();
	const char *dbName() const { return mDbNameValid ? mDbName.c_str() : NULL; }
	const char *canonicalName() const { return mDbNameValid ? mCanonicalName.c_str() : NULL; }
    const CssmNetAddress *dbLocation() const { return mDbLocation; }
    bool operator <(const DbName &other) const
    {
		// invalid is always smaller than valid
		if (!mDbNameValid || !other.mDbNameValid)
			return mDbNameValid < other.mDbNameValid;
	
        // If mDbNames are not equal return whether our mDbName is less than others mDbName.
        if (canonicalName() != other.canonicalName())
            return mDbName < other.mDbName;

        // DbNames are equal so check for pointer equality of DbLocations
        if (mDbLocation == other.mDbLocation)
            return false;

        // If either DbLocations is nil the one that is nil is less than the other.
        if (mDbLocation == nil || other.mDbLocation == nil)
            return mDbLocation < other.mDbLocation;

        // Return which mDbLocation is smaller.
        return *mDbLocation < *other.mDbLocation;
    }
	bool operator ==(const DbName &other) const
	{ return (!(*this < other)) && (!(other < *this)); }
	bool operator !=(const DbName &other) const
	{ return *this < other || other < *this; }

private:
	void CanonicalizeName();

    string mDbName;
	string mCanonicalName;
	bool mDbNameValid;
    CssmNetAddress *mDbLocation;
};


namespace DataWalkers
{

template<class Action>
CssmNetAddress *walk(Action &operate, CssmNetAddress * &addr)
{
    operate(addr);
    walk(operate, addr->Address);
    return addr;
}

} // end namespace DataWalkers

} // end namespace Security

#ifdef _CPP_DBNAME
# pragma export off
#endif

#endif //_DBNAME_H_
