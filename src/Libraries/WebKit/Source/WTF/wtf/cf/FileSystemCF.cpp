/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 31, 2023.
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
#include "config.h"
#include <wtf/FileSystem.h>

#include <CoreFoundation/CFString.h>
#include <wtf/RetainPtr.h>
#include <wtf/text/CString.h>
#include <wtf/text/WTFString.h>

namespace WTF {

CString FileSystem::fileSystemRepresentation(const String& path)
{
    RetainPtr<CFStringRef> cfString = path.createCFString();

    if (!cfString)
        return CString();

    CFIndex size = CFStringGetMaximumSizeOfFileSystemRepresentation(cfString.get());

    Vector<char> buffer(size);

    if (!CFStringGetFileSystemRepresentation(cfString.get(), buffer.data(), buffer.size())) {
        LOG_ERROR("Failed to get filesystem representation to create CString from cfString");
        return CString();
    }

    return buffer.data();
}

String FileSystem::stringFromFileSystemRepresentation(const char* fileSystemRepresentation)
{
    return adoptCF(CFStringCreateWithFileSystemRepresentation(kCFAllocatorDefault, fileSystemRepresentation)).get();
}

RetainPtr<CFURLRef> FileSystem::pathAsURL(const String& path)
{
    return adoptCF(CFURLCreateWithFileSystemPath(nullptr, path.createCFString().get(), kCFURLPOSIXPathStyle, FALSE));
}

} // namespace WTF
