/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 7, 2025.
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
 * June 1, 2001			Allan Nathanson <ajn@apple.com>
 * - public API conversion
 *
 * April 14, 2000		Allan Nathanson <ajn@apple.com>
 * - initial revision
 */

#include <fcntl.h>
#include <paths.h>
#include <unistd.h>

#include "configd.h"
#include "configd_server.h"
#include "session.h"
#include "plugin_support.h"


#define	SNAPSHOT_PATH_STATE	_PATH_VARTMP "configd-state"
#define	SNAPSHOT_PATH_STORE	_PATH_VARTMP "configd-store.plist"
#define	SNAPSHOT_PATH_PATTERN	_PATH_VARTMP "configd-pattern.plist"


#define N_QUICK	100

static CF_RETURNS_RETAINED CFDictionaryRef
_expandStore(CFDictionaryRef storeData)
{
	const void *		keys_q[N_QUICK];
	const void **		keys		= keys_q;
	CFIndex			nElements;
	CFDictionaryRef		newStoreData	= NULL;
	const void *		nValues_q[N_QUICK];
	const void **		nValues		= nValues_q;
	const void *		oValues_q[N_QUICK];
	const void **		oValues		= oValues_q;

	nElements = CFDictionaryGetCount(storeData);
	if (nElements > 0) {
		CFIndex	i;

		if (nElements > (CFIndex)(sizeof(keys_q) / sizeof(CFTypeRef))) {
			keys    = CFAllocatorAllocate(NULL, nElements * sizeof(CFTypeRef), 0);
			oValues = CFAllocatorAllocate(NULL, nElements * sizeof(CFTypeRef), 0);
			nValues = CFAllocatorAllocate(NULL, nElements * sizeof(CFTypeRef), 0);
		}
		memset(nValues, 0, nElements * sizeof(CFTypeRef));

		CFDictionaryGetKeysAndValues(storeData, keys, oValues);
		for (i = 0; i < nElements; i++) {
			CFDataRef		data;

			data = CFDictionaryGetValue(oValues[i], kSCDData);
			if (data) {
				CFPropertyListRef	plist;

				nValues[i] = CFDictionaryCreateMutableCopy(NULL, 0, oValues[i]);

				if (!_SCUnserialize(&plist, data, NULL, 0)) {
					SC_log(LOG_NOTICE, "_SCUnserialize() failed, key=%@", keys[i]);
					continue;
				}

				CFDictionarySetValue((CFMutableDictionaryRef)nValues[i],
						     kSCDData,
						     plist);
				CFRelease(plist);
			} else {
				nValues[i] = CFRetain(oValues[i]);
			}
		}
	}

	newStoreData = CFDictionaryCreate(NULL,
				     keys,
				     nValues,
				     nElements,
				     &kCFTypeDictionaryKeyCallBacks,
				     &kCFTypeDictionaryValueCallBacks);

	if (nElements > 0) {
		CFIndex	i;

		for (i = 0; i < nElements; i++) {
			CFRelease(nValues[i]);
		}

		if (keys != keys_q) {
			CFAllocatorDeallocate(NULL, keys);
			CFAllocatorDeallocate(NULL, oValues);
			CFAllocatorDeallocate(NULL, nValues);
		}
	}

	return newStoreData;
}


__private_extern__
int
__SCDynamicStoreSnapshot(SCDynamicStoreRef store)
{
#pragma unused(store)
	CFDictionaryRef			expandedStoreData;
	FILE				*f;
	int				fd;
	CFDataRef			xmlData;

	/* Save a snapshot of configd's "state" */

	(void) unlink(SNAPSHOT_PATH_STATE);
	fd = open(SNAPSHOT_PATH_STATE, O_WRONLY|O_CREAT|O_TRUNC|O_EXCL, 0644);
	if (fd == -1) {
		return kSCStatusFailed;
	}
	f = fdopen(fd, "w");
	if (f == NULL) {
		return kSCStatusFailed;
	}
	SCPrint(TRUE, f, CFSTR("Main [plug-in] thread :\n\n"));
	SCPrint(TRUE, f, CFSTR("%@\n"), CFRunLoopGetCurrent());
	listSessions(f);
	(void) fclose(f);

	/* Save a snapshot of the "store" data */

	(void) unlink(SNAPSHOT_PATH_STORE);
	fd = open(SNAPSHOT_PATH_STORE, O_WRONLY|O_CREAT|O_TRUNC|O_EXCL, 0644);
	if (fd == -1) {
		return kSCStatusFailed;
	}

	expandedStoreData = _expandStore(storeData);
	xmlData = CFPropertyListCreateData(NULL, expandedStoreData, kCFPropertyListXMLFormat_v1_0, 0, NULL);
	CFRelease(expandedStoreData);
	if (xmlData == NULL) {
		SC_log(LOG_NOTICE, "CFPropertyListCreateData() failed");
		close(fd);
		return kSCStatusFailed;
	}
	(void) write(fd, CFDataGetBytePtr(xmlData), CFDataGetLength(xmlData));
	(void) close(fd);
	CFRelease(xmlData);

	/* Save a snapshot of the "pattern" data */

	(void) unlink(SNAPSHOT_PATH_PATTERN);
	fd = open(SNAPSHOT_PATH_PATTERN, O_WRONLY|O_CREAT|O_TRUNC|O_EXCL, 0644);
	if (fd == -1) {
		return kSCStatusFailed;
	}

	xmlData = CFPropertyListCreateData(NULL, patternData, kCFPropertyListXMLFormat_v1_0, 0, NULL);
	if (xmlData == NULL) {
		SC_log(LOG_NOTICE, "CFPropertyListCreateData() failed");
		close(fd);
		return kSCStatusFailed;
	}
	(void) write(fd, CFDataGetBytePtr(xmlData), CFDataGetLength(xmlData));
	(void) close(fd);
	CFRelease(xmlData);

	return kSCStatusOK;
}


__private_extern__
kern_return_t
_snapshot(mach_port_t server, int *sc_status)
{
	serverSessionRef	mySession;

	mySession = getSession(server);
	if (mySession == NULL) {
		/* you must have an open session to play */
		return kSCStatusNoStoreSession;
	}

	if (!hasRootAccess(mySession)) {
		return kSCStatusAccessError;
	}

	*sc_status = __SCDynamicStoreSnapshot(mySession->store);
	return KERN_SUCCESS;
}
