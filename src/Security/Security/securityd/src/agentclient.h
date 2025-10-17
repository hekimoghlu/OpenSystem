/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 8, 2025.
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
//  agentclient.h
//  securityd
//
//  Created by cschmidt on 11/24/14.
//

#ifndef securityd_agentclient_h
#define securityd_agentclient_h

namespace SecurityAgent {
	enum Reason {
		noReason = 0,					// no reason (not used, used as a NULL)
		unknownReason,					// something else (catch-all internal error)

		// reasons for asking for a new passphrase
		newDatabase = 11,				// need passphrase for a new database
		changePassphrase,				// changing passphrase for existing database

		// reasons for retrying an unlock query
		invalidPassphrase = 21,			// passphrase was wrong

		// reasons for retrying a new passphrase query
		passphraseIsNull = 31,			// empty passphrase
		passphraseTooSimple,			// passphrase is not complex enough
		passphraseRepeated,				// passphrase was used before (must use new one)
		passphraseUnacceptable,			// passphrase unacceptable for some other reason
		oldPassphraseWrong,				// the old passphrase given is wrong

		// reasons for retrying an authorization query
		userNotInGroup = 41,			// authenticated user not in needed group
		unacceptableUser,				// authenticated user unacceptable for some other reason

		// reasons for canceling a staged query
		tooManyTries = 61,				// too many failed attempts to get it right
		noLongerNeeded,					// the queried item is no longer needed
		keychainAddFailed,				// the requested itemed couldn't be added to the keychain
		generalErrorCancel,				// something went wrong so we have to give up now
		resettingPassword,              // The user has indicated that they wish to reset their password

		worldChanged = 101
	};

	typedef enum {
		tool = 'TOOL',
		bundle = 'BNDL',
		unknown = 'UNKN'
	} RequestorType;
}
#endif
