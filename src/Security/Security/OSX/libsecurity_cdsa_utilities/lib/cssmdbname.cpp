/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 25, 2024.
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
#include <security_cdsa_utilities/cssmdbname.h>
#include <security_cdsa_utilities/cssmbridge.h>
#include <security_utilities/utilities.h>

CssmNetAddress::CssmNetAddress(CSSM_DB_RECORDTYPE inAddressType, const CssmData &inAddress)
{
    AddressType = inAddressType;
    Address.Length = inAddress.Length;
    if (Address.Length > 0)
    {
        Address.Data = new uint8[Address.Length];
        memcpy (Address.Data, inAddress.Data, Address.Length);
    }
    else
        Address.Data = NULL;
}

CssmNetAddress::CssmNetAddress(const CSSM_NET_ADDRESS &other)
{
    AddressType = other.AddressType;
    Address.Length = other.Address.Length;
    if (Address.Length > 0)
    {
        Address.Data = new uint8[Address.Length];
        memcpy (Address.Data, other.Address.Data, Address.Length);
    }
    else
        Address.Data = NULL;
}

CssmNetAddress::~CssmNetAddress()
{
    if (Address.Length > 0)
        delete Address.Data;
}

void DbName::CanonicalizeName()
{
	if (mDbNameValid)
	{
		char* s = cached_realpath(mDbName.c_str(), NULL);
		if (s != NULL)
		{
			mCanonicalName = s;
			free(s);
		}
		else
		{
			// the most likely situation here is that the file doesn't exist.
			// we will pull the path apart and try again.
			
			// search backward for the delimiter
			ptrdiff_t n = mDbName.length() - 1;
			
			// all subpaths must be tested, because there may be more than just
			// the file name that doesn't exist.
			while (n > 0)
			{
				while (n > 0 && mDbName[n] != '/') // if the delimiter is 0, we would never
												   // have gotten here in the first place
				{
					n -= 1;
				}
				
				if (n > 0)
				{
					string tmpPath = mDbName.substr(0, n);
					s = cached_realpath(tmpPath.c_str(), NULL);
					if (s != NULL)
					{
						mCanonicalName = s;
						free(s);
						mCanonicalName += mDbName.substr(n, mDbName.length() - n);
						return;
					}
				}
				
				n -= 1;
			}
			
			// if we get here, all other paths have failed.  Just reuse the original string.
			mCanonicalName = mDbName;
		}
	}
}



DbName::DbName(const char *inDbName, const CSSM_NET_ADDRESS *inDbLocation)
	: mDbName(inDbName ? inDbName : ""), mDbNameValid(inDbName), mDbLocation(NULL)
{
    if (inDbLocation)
    {
        mDbLocation = new CssmNetAddress(*inDbLocation);
    }
	
	CanonicalizeName();
}

DbName::DbName(const DbName &other)
	: mDbName(other.mDbName), mDbNameValid(other.mDbNameValid), mDbLocation(NULL)
{
    if (other.mDbLocation)
    {
        mDbLocation = new CssmNetAddress(*other.mDbLocation);
    }
	
	CanonicalizeName();
}

DbName &
DbName::operator =(const DbName &other)
{
	mDbName = other.mDbName;
    mCanonicalName = other.mCanonicalName;
	mDbNameValid = other.mDbNameValid;
    if (other.mDbLocation)
    {
        mDbLocation = new CssmNetAddress(*other.mDbLocation);
    }

	return *this;
}

DbName::~DbName()
{
	delete mDbLocation;
}
