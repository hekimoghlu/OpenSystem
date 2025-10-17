/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 10, 2025.
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
//
// adornment - generic attached-storage facility
//
#include "adornments.h"
#include <security_utilities/utilities.h>
#include <security_utilities/debugging.h>


namespace Security {


//
// Adornment needs a virtual destructor for safe deletion.
//
Adornment::~Adornment()
{ }


//
// Adornable deletes all pointed-to objects when it dies.
//
Adornable::~Adornable()
{
	clearAdornments();
}


//
// Primitive (non-template) adornment operations
//
Adornment *Adornable::getAdornment(Key key) const
{
	if (mAdornments) {
		AdornmentMap::const_iterator it = mAdornments->find(key);
		return (it == mAdornments->end()) ? NULL : it->second;
	} else
		return NULL;	// nada
}

void Adornable::setAdornment(Key key, Adornment *ad)
{
	Adornment *&slot = adornmentSlot(key);
	delete slot;
	slot = ad;
}

Adornment *Adornable::swapAdornment(Key key, Adornment *ad)
{
	std::swap(ad, adornmentSlot(key));
	return ad;
}

Adornment *&Adornable::adornmentSlot(Key key)
{
	if (!mAdornments)
		mAdornments = new AdornmentMap;
	return (*mAdornments)[key];
}

void Adornable::clearAdornments()
{
	if (mAdornments) {
		for_each_map_delete(mAdornments->begin(), mAdornments->end());
		delete mAdornments;
		mAdornments = NULL;
	}
}

}	// end namespace Security
