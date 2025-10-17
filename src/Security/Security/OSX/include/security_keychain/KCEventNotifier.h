/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 25, 2022.
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
 *  KCEventNotifier.h -- OS X CF Notifier for Keychain Events
 */
#ifndef _SECURITY_KCEVENTNOTIFIER_H_
#define _SECURITY_KCEVENTNOTIFIER_H_

#include <CoreFoundation/CFNotificationCenter.h>
#include <CoreFoundation/CFString.h>
#include <security_keychain/Item.h>
#include <securityd_client/dictionary.h>
#include <list>

namespace Security
{

namespace KeychainCore
{

class Keychain;

class KCEventNotifier
{
public:
	static void PostKeychainEvent(SecKeychainEvent kcEvent, 
								  const Keychain& keychain, 
								  const Item &item = Item());
	static void PostKeychainEvent(SecKeychainEvent kcEvent, 
								  const DLDbIdentifier &dlDbIdentifier = DLDbIdentifier(), 
								  const PrimaryKey &primaryKey = PrimaryKey());
};

} // end namespace KeychainCore

} // end namespace Security

#endif /* _SECURITY_KCEVENTNOTIFIER_H_ */
