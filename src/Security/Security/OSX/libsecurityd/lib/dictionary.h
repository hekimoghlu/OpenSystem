/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 16, 2022.
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
#ifndef _DICTIONARY_H__
#define _DICTIONARY_H__


#include <vector>
#include <security_cdsa_utilities/cssmdb.h>

namespace Security {



#define PID_KEY				'pidk'
#define ITEM_KEY			'item'
#define SSUID_KEY			'ssui'
#define IDENTITY_KEY		'idnt'
#define DB_NAME				'dbnm'
#define DB_LOCATION			'dblc'



class NameValuePair
{
protected:
	uint32 mName;
	CssmData mValue;

	CssmData CloneData (const CssmData &value);

public:
	NameValuePair (uint32 name, const CssmData &value);
	NameValuePair (const CssmData &data);
	~NameValuePair ();

	uint32 Name () {return mName;}
	const CssmData& Value () const {return mValue;}
	void Export (CssmData &data) const;
};



typedef std::vector<NameValuePair*> NameValuePairVector;



class NameValueDictionary
{
protected:
	NameValuePairVector mVector;

	int FindPositionByName (uint32 name) const;

	void MakeFromData(const CssmData &data);

public:
	NameValueDictionary ();
	~NameValueDictionary ();
	NameValueDictionary (const CssmData &data);

	void Insert (NameValuePair* pair);
	void RemoveByName (uint32 name);
	const NameValuePair* FindByName (uint32 name) const;

	int CountElements () const;
	const NameValuePair* GetElement (int which);
	void Export (CssmData &data);

	// utility functions
	static void MakeNameValueDictionaryFromDLDbIdentifier (const DLDbIdentifier &identifier, NameValueDictionary &nvd);
	static DLDbIdentifier MakeDLDbIdentifierFromNameValueDictionary (const NameValueDictionary &nvd);
};


};

#endif
