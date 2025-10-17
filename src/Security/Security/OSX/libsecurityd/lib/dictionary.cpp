/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 29, 2024.
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
#include "dictionary.h"
#include <ctype.h>
#include <syslog.h>

namespace Security {

static uint32_t GetUInt32(unsigned char*& finger)
{
	uint32 result = 0;
	unsigned i;
    
	for (i = 0; i < sizeof(uint32); ++i)
	{
		result = (result << 8) | *finger++;
	}
    
	return result;
}



CssmData NameValuePair::CloneData (const CssmData &value)
{
	void* clonedData = (void*) new unsigned char [value.length ()];
	if (clonedData != NULL)
	{
		memcpy (clonedData, value.data (), value.length ());
		return CssmData (clonedData, value.length ());
	}
	else
	{
		CssmError::throwMe(CSSM_ERRCODE_INTERNAL_ERROR);
	}
}



NameValuePair::NameValuePair (uint32 name, const CssmData &value) : mName (name), mValue (CloneData (value))
{
}



NameValuePair::NameValuePair (const CssmData &data)
{
    size_t actualLength = data.length();
    
    if (actualLength < (2 * sizeof(uint32_t))) {
        CssmError::throwMe(CSSM_ERRCODE_INTERNAL_ERROR);
    }
    
	// the first four bytes are the name
	unsigned char* finger = (unsigned char*) data.data ();
	mName = GetUInt32(finger);
	uint32 computedLength = GetUInt32(finger);
    
    if (actualLength != ((2 * sizeof(uint32_t)) + computedLength)) {
        CssmError::throwMe(CSSM_ERRCODE_INTERNAL_ERROR);
    }
        
	// what's left is the data
	mValue = CloneData (CssmData (finger, computedLength));
}



NameValuePair::~NameValuePair ()
{
	delete[] (unsigned char*) mValue.data ();
}



void NameValuePair::Export (CssmData &data) const
{
	// export the data in the format name length data
	size_t outSize = 2 * sizeof (uint32) + mValue.length ();
	unsigned char* d = (unsigned char*) malloc(outSize);
	unsigned char* finger = d;
	
	// export the name
	uint32 intBuffer = mName;

	int i;
	for (i = sizeof (uint32) - 1; i >= 0; --i)
	{
		finger[i] = intBuffer & 0xFF;
		intBuffer >>= 8;
	}
	
	// export the length
	finger += sizeof (uint32);
	intBuffer = (uint32)mValue.length ();
	for (i = sizeof (uint32) - 1; i >= 0; --i)
	{
		finger[i] = intBuffer & 0xFF;
		intBuffer >>= 8;
	}

	// export the data
	finger += sizeof (uint32);
	memcpy (finger, mValue.data (), mValue.length ());
	
	data = CssmData (d, outSize);
}



NameValueDictionary::NameValueDictionary ()
{
}



NameValueDictionary::~NameValueDictionary ()
{
	// to prevent leaks, delete all members of the vector
	size_t i = mVector.size ();
	while (i > 0)
	{
		delete mVector[--i];
		
		mVector.erase (mVector.begin () + i);
	}
}



// To work around 5964438, move code out of the constructor
void NameValueDictionary::MakeFromData(const CssmData &data)
{
	// reconstruct a name value dictionary from a series of exported NameValuePair blobs
	unsigned char* finger = (unsigned char*) data.data ();
	unsigned char* target = finger + data.length ();
    size_t running_length_total = 0;
    
    // rdar://76759018
    if (data.length() < (2 * sizeof (uint32))) {
        return;
    }
    
	bool done = false;

	do
	{
		// compute the length of data blob
		unsigned int i;
		uint32 length = 0;
		for (i = sizeof (uint32); i < 2 * sizeof (uint32); ++i) {
			length = (length << 8) | finger[i];
		}
		
		if (length > data.length() || ((running_length_total + length) > data.length())) {
			break;
		}
		
		// add the length of the "header"
		length += 2 * sizeof (uint32);

        if (length > data.length() || ((running_length_total + length) > data.length())) {
            break;
        }

        running_length_total += length;

		// do some integrity checking on the data.
		uint32 itemLength = 0;
		unsigned char* fingerX = finger;
		
		// extract the name in a printable format
		char nameBuff[5];
		char* nameFinger = nameBuff;

		// work around a bug with invalid lengths coming from securityd
		if (fingerX + sizeof(uint32) < target)
		{
			*nameFinger++ = (char) *fingerX++;
			*nameFinger++ = (char) *fingerX++;
			*nameFinger++ = (char) *fingerX++;
			*nameFinger++ = (char) *fingerX++;
			*nameFinger++ = 0;
			
			itemLength = GetUInt32(fingerX);
			
			if (fingerX + itemLength > target) // this is the bug
			{
                done = true;
			}
		}
          
        // This shouldn't crash any more...
		Insert (new NameValuePair (CssmData (finger, length)));
		
		// skip to the next data
		finger += length;
	} while (!done && finger < target);
}



NameValueDictionary::NameValueDictionary (const CssmData &data)
{
	MakeFromData(data);
}
	


void NameValueDictionary::Insert (NameValuePair* pair)
{
	mVector.push_back (pair);
}



void NameValueDictionary::RemoveByName (uint32 name)
{
	int which = FindPositionByName (name);
	if (which != -1)
	{
		NameValuePair* nvp = mVector[which];
		mVector.erase (mVector.begin () + which);
		delete nvp;
	}
}



int NameValueDictionary::FindPositionByName (uint32 name) const
{
	int target = CountElements ();
	int i;
	
	for (i = 0; i < target; ++i)
	{
		if (mVector[i]->Name () == name)
		{
			return i;
		}
	}
	
	return -1;
}



const NameValuePair* NameValueDictionary::FindByName (uint32 name) const
{
	int which = FindPositionByName (name);
	return which == -1 ? NULL : mVector[which];
}




int NameValueDictionary::CountElements () const
{
	return (int)mVector.size ();
}



const NameValuePair* NameValueDictionary::GetElement (int which)
{
	return mVector[which];
}



void NameValueDictionary::Export (CssmData &outData)
{
	// get each element in the dictionary, and add it to the data blob
	int i;
	uint32 length = 0;
	unsigned char* data = 0;

	for (i = 0; i < CountElements (); ++i)
	{
		CssmData exportedData;
		const NameValuePair *nvp = GetElement (i);
		nvp->Export (exportedData);
		
		uint32 oldLength = length;
		length += exportedData.length ();
		data = (unsigned char*) realloc (data, length);
		
		memcpy (data + oldLength, exportedData.data (), exportedData.length ());
		
		free(exportedData.data());
	}
	
	outData = CssmData (data, length);
}



void NameValueDictionary::MakeNameValueDictionaryFromDLDbIdentifier (const DLDbIdentifier &identifier, NameValueDictionary &nvd)
{
	// get the subserviceID
	DLDbIdentifier d = identifier;
	
	const CssmSubserviceUid &ssuid = identifier.ssuid ();
	CSSM_SUBSERVICE_UID baseID = ssuid;
	baseID.Version.Major = h2n (baseID.Version.Major);
	baseID.Version.Minor = h2n (baseID.Version.Minor);
	baseID.SubserviceId = h2n (baseID.SubserviceId);
	baseID.SubserviceType = h2n (baseID.SubserviceType);
	
	nvd.Insert (new NameValuePair (SSUID_KEY, CssmData::wrap(baseID)));
	
	// get the name
	const char* dbName = identifier.dbName ();
	if (dbName != NULL)
	{
		nvd.Insert (new NameValuePair (DB_NAME, CssmData::wrap (dbName, strlen (dbName) + 1)));
	}
	
	// get the net address
	const CSSM_NET_ADDRESS* add = identifier.dbLocation ();
	if (add != NULL)
	{
		nvd.Insert (new NameValuePair (DB_LOCATION, CssmData::wrap (add)));
	}
}



DLDbIdentifier NameValueDictionary::MakeDLDbIdentifierFromNameValueDictionary (const NameValueDictionary &nvd)
{
	/*
		According to the code in MakeNameValueDictionaryFromDLDbIdentifier, SSUID_KEY
		is required, but both DB_NAME and DB_LOCATION are allowed to be missing. In 
		all of these cases, it is possible that FindByName returns NULL.
	*/
	
	const NameValuePair *nvpSSUID = nvd.FindByName (SSUID_KEY);
	if (nvpSSUID == NULL)
		CssmError::throwMe(CSSM_ERRCODE_INTERNAL_ERROR);
		
	CSSM_SUBSERVICE_UID* uid = (CSSM_SUBSERVICE_UID*) nvpSSUID->Value ().data ();
	if (uid == NULL)
	{
		CssmError::throwMe(CSSM_ERRCODE_INTERNAL_ERROR);
	}
	
	CSSM_SUBSERVICE_UID baseID = *uid;
	
	baseID.Version.Major = n2h (baseID.Version.Major);
	baseID.Version.Minor = n2h (baseID.Version.Minor);
	baseID.SubserviceId = n2h (baseID.SubserviceId);
	baseID.SubserviceType = n2h (baseID.SubserviceType);
	
	const NameValuePair *nvpDBNAME = nvd.FindByName (DB_NAME);
	char* name = nvpDBNAME ? (char*) nvpDBNAME->Value ().data () : NULL;
	
	const NameValuePair* nvp = nvd.FindByName (DB_LOCATION);
	CSSM_NET_ADDRESS* address = nvp ? (CSSM_NET_ADDRESS*) nvp->Value ().data () : NULL;
	
	return DLDbIdentifier (baseID, name, address);
}

}; // end Security namespace
