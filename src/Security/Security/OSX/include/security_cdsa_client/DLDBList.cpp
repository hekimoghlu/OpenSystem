/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 25, 2023.
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
    DLDbList.cpp
*/

#include "DLDBList.h"

using namespace CssmClient;

//----------------------------------------------------------------------
//			DLDbList implementation
//----------------------------------------------------------------------

void DLDbList::add(const DLDbIdentifier& dldbIdentifier)	// Adds at end if not in list
{
    for (DLDbList::const_iterator ix=begin();ix!=end();ix++)
        if (*ix==dldbIdentifier)		// already in list
            return;
    push_back(dldbIdentifier);
    changed(true);
}

void DLDbList::remove(const DLDbIdentifier& dldbIdentifier)	// Removes from list
{
    for (DLDbList::iterator ix=begin();ix!=end();ix++)
	if (*ix==dldbIdentifier)		// found in list
	{
		erase(ix);
		changed(true);
		break;
	}
}

void DLDbList::save()
{
}
