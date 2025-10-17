/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 2, 2024.
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
#ifndef _SCPREFERENCESINTERNAL_H
#define _SCPREFERENCESINTERNAL_H

#include <dispatch/dispatch.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <os/log.h>
#include <os/state_private.h>
#include <CoreFoundation/CoreFoundation.h>
#include <CoreFoundation/CFRuntime.h>

#ifndef	SC_LOG_HANDLE
#define	SC_LOG_HANDLE	__log_SCPreferences
#endif	// SC_LOG_HANDLE
#include <SystemConfiguration/SystemConfiguration.h>
#include <SystemConfiguration/SCValidation.h>
#include <SystemConfiguration/SCPrivate.h>

#include <SystemConfiguration/SCPreferences.h>
#include <SystemConfiguration/SCDynamicStore.h>


#define PREFS_DEFAULT_DIR_PATH_RELATIVE	"Library/Preferences/SystemConfiguration"
#define	PREFS_DEFAULT_DIR_RELATIVE	CFSTR(PREFS_DEFAULT_DIR_PATH_RELATIVE "/")

#define	PREFS_DEFAULT_DIR_PATH		"/" PREFS_DEFAULT_DIR_PATH_RELATIVE
#define	PREFS_DEFAULT_DIR		CFSTR(PREFS_DEFAULT_DIR_PATH)

#define	PREFS_DEFAULT_CONFIG_PLIST	"preferences.plist"
#define	PREFS_DEFAULT_CONFIG		CFSTR(PREFS_DEFAULT_CONFIG_PLIST)

#define	PREFS_DEFAULT_USER_DIR		CFSTR("Library/Preferences")

#define	INTERFACES_DEFAULT_CONFIG_PLIST	"NetworkInterfaces.plist"
#define	INTERFACES_DEFAULT_CONFIG	CFSTR(INTERFACES_DEFAULT_CONFIG_PLIST)
#define	INTERFACES			CFSTR("Interfaces")


/* Define the per-preference-handle structure */
typedef struct {

	/* base CFType information */
	CFRuntimeBase		cfBase;

	/* lock */
	pthread_mutex_t		lock;

	/* session name */
	CFStringRef		name;

	/* preferences ID */
	CFStringRef		prefsID;

	/* options */
	CFDictionaryRef		options;

	/* configuration file */
	char			*path;

	/* preferences lock, lock file */
	Boolean			locked;
	int			lockFD;
	char			*lockPath;
	struct timeval		lockTime;

	/* configuration file signature */
	CFDataRef		signature;

	/* configd session */
	SCDynamicStoreRef	session;
	SCDynamicStoreRef	sessionNoO_EXLOCK;
	int			sessionRefcnt;

	/* configd session keys */
	CFStringRef		sessionKeyLock;
	CFStringRef		sessionKeyCommit;
	CFStringRef		sessionKeyApply;

	/* run loop source, callout, context, rl scheduling info */
	Boolean			scheduled;
	CFRunLoopSourceRef      rls;
	SCPreferencesCallBack	rlsFunction;
	SCPreferencesContext	rlsContext;
	CFMutableArrayRef       rlList;
	dispatch_queue_t	dispatchQueue;		// SCPreferencesSetDispatchQueue

	/* preferences */
	CFMutableDictionaryRef	prefs;

	/* companion preferences, manipulate under lock */
	SCPreferencesRef	parent;		// [strong] reference from companion to parent
	CFMutableDictionaryRef	companions;	// [weak] reference from parent to companions

	/* flags */
	Boolean			accessed;
	Boolean			changed;
	Boolean			isRoot;
	uint32_t		nc_flags;	// SCNetworkConfiguration flags

	/* authorization, helper */
	CFDataRef		authorizationData;
	mach_port_t		helper_port;

} SCPreferencesPrivate, *SCPreferencesPrivateRef;


/* Define signature data */
typedef struct {
	int64_t		st_dev;		/* inode's device */
	uint64_t	st_ino;		/* inode's number */
	uint64_t	tv_sec;		/* time of last data modification */
	uint64_t	tv_nsec;
	off_t		st_size;	/* file size, in bytes */
} SCPSignatureData, *SCPSignatureDataRef;


__BEGIN_DECLS

static __inline__ CFTypeRef
isA_SCPreferences(CFTypeRef obj)
{
	return (isA_CFType(obj, SCPreferencesGetTypeID()));
}

os_log_t
__log_SCPreferences			(void);

Boolean
__SCPreferencesCreate_helper		(SCPreferencesRef	prefs);

void
__SCPreferencesAccess			(SCPreferencesRef	prefs);

void
__SCPreferencesAddSessionKeys		(SCPreferencesRef       prefs);

Boolean
__SCPreferencesAddSession		(SCPreferencesRef       prefs);

Boolean
__SCPreferencesIsEmpty			(SCPreferencesRef	prefs);

void
__SCPreferencesRemoveSession		(SCPreferencesRef       prefs);

void
__SCPreferencesUpdateLockedState	(SCPreferencesRef       prefs,
					 Boolean		locked);

CF_RETURNS_RETAINED
CFDataRef
__SCPSignatureFromStatbuf		(const struct stat	*statBuf);

char *
__SCPreferencesPath			(CFAllocatorRef		allocator,
					 CFStringRef		prefsID,
					 Boolean 		usePrebootVolume);

off_t
__SCPreferencesPrefsSize		(SCPreferencesRef	prefs);

CF_RETURNS_RETAINED
CFStringRef
_SCPNotificationKey			(CFAllocatorRef		allocator,
					 CFStringRef		prefsID,
					 int			keyType);

uint32_t
__SCPreferencesGetNetworkConfigurationFlags
					(SCPreferencesRef	prefs);

void
__SCPreferencesSetNetworkConfigurationFlags
					(SCPreferencesRef	prefs,
					 uint32_t		nc_flags);

Boolean
__SCPreferencesUsingDefaultPrefs	(SCPreferencesRef	prefs);

__private_extern__
void
__SCPreferencesHandleInternalStatus	(uint32_t		*sc_status);

__END_DECLS

#endif /* _SCPREFERENCESINTERNAL_H */
