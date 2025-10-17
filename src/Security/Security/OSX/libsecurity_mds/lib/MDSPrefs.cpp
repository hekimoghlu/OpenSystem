/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 1, 2022.
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
// MDSPrefs.cpp
//

#include "MDSPrefs.h"
#include <CoreFoundation/CFPreferences.h>
#include <stdlib.h>

// Construct the preferences object and read the current preference settings.

MDSPrefs::MDSPrefs()
    :	mPluginFolders(NULL)
{
    if (!readPathFromPrefs() && !readPathFromEnv())
        useDefaultPath();
}

// Destroy the preferences object.

MDSPrefs::~MDSPrefs()
{
    if (mPluginFolders)
        CFRelease(mPluginFolders);
}

// Obtain the plugin path from a preferences file. Returns true on success of false
// if no prefs could be found.

bool
MDSPrefs::readPathFromPrefs()
{
    static const CFStringRef kPrefsSuite = CFSTR("com.apple.mds");
    static const CFStringRef kPluginPathKey = CFSTR("securityPluginPath");

    bool result = true;

    CFPreferencesAddSuitePreferencesToApp(kCFPreferencesCurrentApplication, kPrefsSuite);
    
    CFPropertyListRef value;
    value = CFPreferencesCopyAppValue(kPluginPathKey, kCFPreferencesCurrentApplication);
    
    if (CFGetTypeID(value) != CFArrayGetTypeID())
		// the prefs object is not an array, so fail
		result = false;
		
	else {
        // make sure that all array elements are strings

		CFArrayRef array = (CFArrayRef) value;
		int numItems = CFArrayGetCount(array);
		for (int i = 0; i < numItems; i++)
			if (CFGetTypeID(CFArrayGetValueAtIndex(array, i)) != CFStringGetTypeID()) {
				result = false;
				break;
			}
	}
	
	if (result)
        mPluginFolders = (CFArrayRef) value;
	else
        CFRelease(value);

    return result;
}

bool
MDSPrefs::readPathFromEnv()
{
    static const char *kPluginPathEnv = "MDSPATH";
    static const CFStringRef kSeparator = CFSTR(":");

    char *envValue = getenv(kPluginPathEnv);
    if (envValue) {
        CFStringRef path = CFStringCreateWithCString(NULL, envValue, kCFStringEncodingUTF8);

        mPluginFolders = CFStringCreateArrayBySeparatingStrings(NULL, path, kSeparator);
        
        CFRelease(path);
        return true;
    }
    
    return false;
}

void
MDSPrefs::useDefaultPath()
{
    static const CFStringRef kDefaultPluginPath = CFSTR("/System/Library/Security");

    mPluginFolders = CFArrayCreate(NULL, (const void **) &kDefaultPluginPath, 1, &kCFTypeArrayCallBacks);
}

// Retrieve the elements of the plugin path.

int
MDSPrefs::getNumberOfPluginFolders() const
{
    if (mPluginFolders)
        return CFArrayGetCount(mPluginFolders);
        
    return 0;
}

const char *
MDSPrefs::getPluginFolder(int index)
{
    if (mPluginFolders) {
        int numValues = CFArrayGetCount(mPluginFolders);
        if (index >= 0 && index < numValues) {
            CFStringRef value = (CFStringRef) CFArrayGetValueAtIndex(mPluginFolders, index);
            if (value) {
                // we have to copy the string since it may be using a different native
                // encoding than the one we want. the copy is put in a temporary buffer,
                // so its lifetime is limited to the next call to getPluginFolder() or to
                // the destruction of the MDSPrefs object. Very long paths will silently fail.
                
                if (CFStringGetCString(value, mTempBuffer, kTempBufferSize, kCFStringEncodingUTF8))
                    return mTempBuffer;
            }
        }
    }
    
    return NULL;
}

