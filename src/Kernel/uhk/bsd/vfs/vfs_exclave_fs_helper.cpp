/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 21, 2025.
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
#include <sys/vnode.h>
extern "C" {
#include <vfs/vfs_exclave_fs.h>
}
#include <IOKit/IOPlatformExpert.h>

#define APFS_VOLUME_OBJECT "AppleAPFSVolume"
#define kAPFSVolGroupUUIDKey "VolGroupUUID"

extern "C"
int
vfs_exclave_fs_query_volume_group(const uuid_string_t vguuid_str, bool *exists)
{
#if XNU_TARGET_OS_OSX
	OSDictionary *target = NULL, *filter = NULL;
	OSString *string = NULL;
	IOService *service = NULL;
	int error = 0;
	uuid_t vguuid;

	*exists = false;

	// Verify input uuid is a valid uuid
	error = uuid_parse(vguuid_str, vguuid);
	if (error) {
		return EINVAL;
	}

	// Look for APFS volume object that has Volume Group that matches the one we're looking for
	target = IOService::serviceMatching(APFS_VOLUME_OBJECT);
	if (!target) {
		// No APFS volumes found?
		return ENXIO;
	}

	filter = OSDictionary::withCapacity(1);
	if (!filter) {
		error = ENOMEM;
		goto out;
	}

	string = OSString::withCStringNoCopy(vguuid_str);
	if (!string) {
		error = ENOMEM;
		goto out;
	}

	if (!filter->setObject(kAPFSVolGroupUUIDKey, string)) {
		error = ENXIO;
		goto out;
	}

	if (!target->setObject(gIOPropertyMatchKey, filter)) {
		error = ENXIO;
		goto out;
	}

	if ((service = IOService::copyMatchingService(target)) != NULL) {
		*exists = true;
	}

out:
	if (target) {
		target->release();
	}

	if (filter) {
		filter->release();
	}

	if (string) {
		string->release();
	}

	if (service) {
		service->release();
	}

	return error;
#else
#pragma unused(vguuid_str)
#pragma unused(exists)
	return ENOTSUP;
#endif
}
