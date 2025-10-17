/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 25, 2024.
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
#include <pthread/pthread.h>
#include <sys/sysctl.h>
#include <SoftLinking/SoftLinking.h>

#include "OSKext.h"
#include "OSKextPrivate.h"

/* Avoid creating dependency cycles, since KernelManagement -> Foundation -> ... -> IOKitUser */
SOFT_LINK_OPTIONAL_FRAMEWORK(Frameworks, KernelManagement);

SOFT_LINK_FUNCTION(KernelManagement, KMLoadExtensionsWithPaths, SOFT_KMLoadExtensionsWithPaths,
                   OSReturn,
                   (CFArrayRef paths, CFArrayRef dependencyAndFolderPaths),
                   (paths, dependencyAndFolderPaths));

SOFT_LINK_FUNCTION(KernelManagement, KMLoadExtensionsWithIdentifiers, SOFT_KMLoadExtensionsWithIdentifiers,
                   OSReturn,
                   (CFArrayRef identifiers, CFArrayRef dependencyAndFolderPaths),
                   (identifiers, dependencyAndFolderPaths));

SOFT_LINK_FUNCTION(KernelManagement, KMUnloadExtensionsWithIdentifiers, SOFT_KMUnloadExtensionsWithIdentifiers,
                   OSReturn,
                   (CFArrayRef identifiers),
                   (identifiers));

SOFT_LINK_FUNCTION(KernelManagement, KMExtensionPathForBundleIdentifier, SOFT_KMExtensionPathForBundleIdentifier,
                   CFStringRef,
                   (CFStringRef identifier),
                   (identifier));

bool shimmingEnabled()
{
	uint32_t backOff = 0; // there's a new sheriff in town
	size_t sizeOfBackOff = sizeof(backOff);
	if (!(sysctlbyname("hw.use_kernelmanagerd", &backOff, &sizeOfBackOff, NULL, 0) == 0 && backOff)) {
		OSKextLog(NULL,
			kOSKextLogErrorLevel | kOSKextLogGeneralFlag,
			"Shimming not enabled - defaulting to legacy behavior.");
		return false;
	}

	if (!isKernelManagementAvailable()) {
		OSKextLog(NULL,
			kOSKextLogErrorLevel | kOSKextLogGeneralFlag,
			"KernelManagement soft link failed - defaulting to legacy behavior.");
		return false;
	}
	return true;
}

CFStringRef kernelmanagement_path_for_bundle_id(CFStringRef identifier)
{
	CFStringRef result = SOFT_KMExtensionPathForBundleIdentifier(identifier);
	return result;
}

OSReturn kernelmanagement_load_kext_url(CFURLRef url, CFArrayRef dependencyKextAndFolderPaths)
{
    CFStringRef path = CFURLCopyFileSystemPath(url, kCFURLPOSIXPathStyle);
    if (!path) {
	return kOSReturnError;
    }

    const void *pathArray[] = { (void *)path };
    CFArrayRef paths = CFArrayCreate(kCFAllocatorDefault, (const void **)&pathArray, 1, &kCFTypeArrayCallBacks);
    if (!paths) {
        return kOSReturnError;
    }

    OSReturn result = SOFT_KMLoadExtensionsWithPaths(paths, dependencyKextAndFolderPaths);

    CFRelease(paths);
    return result;
}

OSReturn kernelmanagement_load_kext_identifier(CFStringRef identifier, CFArrayRef dependencyKextAndFolderPaths)
{
    const void *identifiersArray[] = { (void *)identifier };
    CFArrayRef identifiers = CFArrayCreate(kCFAllocatorDefault, (const void **)&identifiersArray, 1, &kCFTypeArrayCallBacks);
    if (!identifiers) {
        return kOSReturnError;
    }

    OSReturn result = SOFT_KMLoadExtensionsWithIdentifiers(identifiers, dependencyKextAndFolderPaths);

    CFRelease(identifiers);
    return result;
}

OSReturn kernelmanagement_unload_kext_identifier(CFStringRef identifier)
{
    const void *identifiersArray[] = { (void *)identifier };
    CFArrayRef identifiers = CFArrayCreate(kCFAllocatorDefault, (const void **)&identifiersArray, 1, &kCFTypeArrayCallBacks);
    if (!identifiers) {
        return kOSReturnError;
    }

    OSReturn result = SOFT_KMUnloadExtensionsWithIdentifiers(identifiers);

    CFRelease(identifiers);
    return result;
}
