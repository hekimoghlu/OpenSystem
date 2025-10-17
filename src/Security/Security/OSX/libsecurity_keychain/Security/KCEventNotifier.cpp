/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 11, 2022.
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
#include <securityd_client/ssclient.h>
#include "KCEventNotifier.h"
#include "KCExceptions.h"
#include "Keychains.h"

using namespace KeychainCore;

void KCEventNotifier::PostKeychainEvent(SecKeychainEvent whichEvent, const Keychain &keychain, const Item &kcItem)
{
	DLDbIdentifier dlDbIdentifier;
	PrimaryKey primaryKey;

	if (keychain)
		dlDbIdentifier = keychain->dlDbIdentifier();

    if (kcItem)
		primaryKey = kcItem->primaryKey();

	PostKeychainEvent(whichEvent, dlDbIdentifier, primaryKey);
}


void KCEventNotifier::PostKeychainEvent(SecKeychainEvent whichEvent,
										const DLDbIdentifier &dlDbIdentifier, 
										const PrimaryKey &primaryKey)
{
	NameValueDictionary nvd;

	Endian<pid_t> thePid = getpid();
	nvd.Insert (new NameValuePair (PID_KEY, CssmData (reinterpret_cast<void*>(&thePid), sizeof (pid_t))));

	if (dlDbIdentifier)
	{
		NameValueDictionary::MakeNameValueDictionaryFromDLDbIdentifier (dlDbIdentifier, nvd);
	}

	CssmData* pKey = primaryKey;
	
    if (primaryKey)
    {
		nvd.Insert (new NameValuePair (ITEM_KEY, *pKey));
    }

	// flatten the dictionary
	CssmData data;
	nvd.Export (data);

    /* enforce a maximum size of 16k for notifications */
    if (data.length() <= 16384) {
        SecurityServer::ClientSession cs (Allocator::standard(), Allocator::standard());
        cs.postNotification (SecurityServer::kNotificationDomainDatabase, whichEvent, data);

        secinfo("kcnotify", "KCEventNotifier::PostKeychainEvent posted event %u", (unsigned int) whichEvent);
    }

	free (data.data ());
}
