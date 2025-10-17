/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 31, 2023.
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
 * Modification History
 *
 * May 1, 2003			Allan Nathanson <ajn@apple.com>
 * - add console [session] information SPIs
 *
 * June 1, 2001			Allan Nathanson <ajn@apple.com>
 * - public API conversion
 *
 * January 2, 2001		Allan Nathanson <ajn@apple.com>
 * - initial revision
 */

#include <SystemConfiguration/SystemConfiguration.h>
#include <SystemConfiguration/SCValidation.h>
#include <SystemConfiguration/SCPrivate.h>


#undef	kSCPropUsersConsoleUserName
#define	kSCPropUsersConsoleUserName	CFSTR("Name")

#undef	kSCPropUsersConsoleUserUID
#define	kSCPropUsersConsoleUserUID	CFSTR("UID")

#undef	kSCPropUsersConsoleUserGID
#define	kSCPropUsersConsoleUserGID	CFSTR("GID")

#undef	kSCPropUsersConsoleSessionInfo
#define	kSCPropUsersConsoleSessionInfo	CFSTR("SessionInfo")


// from CoreGraphics (CGSession.h)
const CFStringRef kSCConsoleSessionUserName		= CFSTR("kCGSSessionUserNameKey");		/* value is CFString */
const CFStringRef kSCConsoleSessionUID			= CFSTR("kCGSSessionUserIDKey");		/* value is CFNumber (a uid_t) */
const CFStringRef kSCConsoleSessionConsoleSet		= CFSTR("kCGSSessionConsoleSetKey");		/* value is CFNumber */
const CFStringRef kSCConsoleSessionOnConsole		= CFSTR("kCGSSessionOnConsoleKey");		/* value is CFBoolean */
const CFStringRef kSCConsoleSessionLoginDone		= CFSTR("kCGSessionLoginDoneKey");		/* value is CFBoolean */

// from CoreGraphics (CGSSession.h)
const CFStringRef kSCConsoleSessionID			= CFSTR("kCGSSessionIDKey");			/* value is CFNumber */

// for loginwindow
const CFStringRef kSCConsoleSessionSystemSafeBoot	= CFSTR("kCGSSessionSystemSafeBoot");		/* value is CFBoolean */
const CFStringRef kSCConsoleSessionLoginwindowSafeLogin	= CFSTR("kCGSSessionLoginwindowSafeLogin");	/* value is CFBoolean */


CFStringRef
SCDynamicStoreKeyCreateConsoleUser(CFAllocatorRef allocator)
{
	return SCDynamicStoreKeyCreate(allocator,
				       CFSTR("%@/%@/%@"),
				       kSCDynamicStoreDomainState,
				       kSCCompUsers,
				       kSCEntUsersConsoleUser);
}


CFStringRef
SCDynamicStoreCopyConsoleUser(SCDynamicStoreRef	store,
			      uid_t		*uid,
			      gid_t		*gid)
{
	CFStringRef		consoleUser	= NULL;
	CFDictionaryRef		dict		= NULL;
	CFStringRef		key;

	key  = SCDynamicStoreKeyCreateConsoleUser(NULL);
	dict = SCDynamicStoreCopyValue(store, key);
	CFRelease(key);
	if (!isA_CFDictionary(dict)) {
		_SCErrorSet(kSCStatusNoKey);
		goto done;
	}

	consoleUser = CFDictionaryGetValue(dict, kSCPropUsersConsoleUserName);
	consoleUser = isA_CFString(consoleUser);
	if (!consoleUser) {
		_SCErrorSet(kSCStatusNoKey);
		goto done;
	}

	CFRetain(consoleUser);

	if (uid) {
		CFNumberRef	num;
		SInt32		val;

		num = CFDictionaryGetValue(dict, kSCPropUsersConsoleUserUID);
		if (isA_CFNumber(num)) {
			if (CFNumberGetValue(num, kCFNumberSInt32Type, &val)) {
				*uid = (uid_t)val;
			}
		}
	}

	if (gid) {
		CFNumberRef	num;
		SInt32		val;

		num = CFDictionaryGetValue(dict, kSCPropUsersConsoleUserGID);
		if (isA_CFNumber(num)) {
			if (CFNumberGetValue(num, kCFNumberSInt32Type, &val)) {
				*gid = (gid_t)val;
			}
		}
	}

    done :

	if (dict)		CFRelease(dict);
	return consoleUser;
}


CFArrayRef
SCDynamicStoreCopyConsoleInformation(SCDynamicStoreRef store)
{
	CFDictionaryRef		dict		= NULL;
	CFArrayRef		info		= NULL;
	CFStringRef		key;

	key  = SCDynamicStoreKeyCreateConsoleUser(NULL);
	dict = SCDynamicStoreCopyValue(store, key);
	CFRelease(key);
	if (!isA_CFDictionary(dict)) {
		_SCErrorSet(kSCStatusNoKey);
		goto done;
	}

	info = CFDictionaryGetValue(dict, kSCPropUsersConsoleSessionInfo);
	info = isA_CFArray(info);
	if (info == NULL) {
		_SCErrorSet(kSCStatusNoKey);
		goto done;
	}

	CFRetain(info);

    done :

	if (dict)		CFRelease(dict);
	return info;
}


Boolean
SCDynamicStoreSetConsoleInformation(SCDynamicStoreRef	store,
				    const char		*user,
				    uid_t		uid,
				    gid_t		gid,
				    CFArrayRef		sessions)
{
	CFStringRef		consoleUser;
	CFMutableDictionaryRef	dict		= NULL;
	CFStringRef		key		= SCDynamicStoreKeyCreateConsoleUser(NULL);
	Boolean			ok		= FALSE;

	if ((user == NULL) && (sessions == NULL)) {
		ok = SCDynamicStoreRemoveValue(store, key);
		goto done;
	}

	dict = CFDictionaryCreateMutable(NULL,
					 0,
					 &kCFTypeDictionaryKeyCallBacks,
					 &kCFTypeDictionaryValueCallBacks);

	if (user != NULL) {
		CFNumberRef	num;

		consoleUser = CFStringCreateWithCString(NULL, user, kCFStringEncodingMacRoman);
		CFDictionarySetValue(dict, kSCPropUsersConsoleUserName, consoleUser);
		CFRelease(consoleUser);

		num = CFNumberCreate(NULL, kCFNumberSInt32Type, (SInt32 *)&uid);
		CFDictionarySetValue(dict, kSCPropUsersConsoleUserUID, num);
		CFRelease(num);

		num = CFNumberCreate(NULL, kCFNumberSInt32Type, (SInt32 *)&gid);
		CFDictionarySetValue(dict, kSCPropUsersConsoleUserGID, num);
		CFRelease(num);
	}

	if (sessions != NULL) {
		CFDictionarySetValue(dict, kSCPropUsersConsoleSessionInfo, sessions);
	}

	ok = SCDynamicStoreSetValue(store, key, dict);

    done :

	if (dict)		CFRelease(dict);
	if (key)		CFRelease(key);
	return ok;
}


Boolean
SCDynamicStoreSetConsoleUser(SCDynamicStoreRef	store,
			     const char		*user,
			     uid_t		uid,
			     gid_t		gid)
{
	CFStringRef		consoleUser;
	CFMutableDictionaryRef	dict		= NULL;
	CFStringRef		key		= SCDynamicStoreKeyCreateConsoleUser(NULL);
	CFNumberRef		num;
	Boolean			ok		= FALSE;

	if (user == NULL) {
		ok = SCDynamicStoreRemoveValue(store, key);
		goto done;
	}

	dict = CFDictionaryCreateMutable(NULL,
					 0,
					 &kCFTypeDictionaryKeyCallBacks,
					 &kCFTypeDictionaryValueCallBacks);

	consoleUser = CFStringCreateWithCString(NULL, user, kCFStringEncodingMacRoman);
	CFDictionarySetValue(dict, kSCPropUsersConsoleUserName, consoleUser);
	CFRelease(consoleUser);

	num = CFNumberCreate(NULL, kCFNumberSInt32Type, (SInt32 *)&uid);
	CFDictionarySetValue(dict, kSCPropUsersConsoleUserUID, num);
	CFRelease(num);

	num = CFNumberCreate(NULL, kCFNumberSInt32Type, (SInt32 *)&gid);
	CFDictionarySetValue(dict, kSCPropUsersConsoleUserGID, num);
	CFRelease(num);

	ok = SCDynamicStoreSetValue(store, key, dict);

    done :

	if (dict)		CFRelease(dict);
	if (key)		CFRelease(key);
	return ok;
}
