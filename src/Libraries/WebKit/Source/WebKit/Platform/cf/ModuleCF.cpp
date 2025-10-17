/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 13, 2025.
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
#include "Module.h"

namespace WebKit {

bool Module::load()
{
    RetainPtr<CFURLRef> bundleURL = adoptCF(CFURLCreateWithFileSystemPath(kCFAllocatorDefault, m_path.createCFString().get(), kCFURLPOSIXPathStyle, FALSE));
    if (!bundleURL)
        return false;

    RetainPtr<CFBundleRef> bundle = adoptCF(CFBundleCreate(kCFAllocatorDefault, bundleURL.get()));
    if (!bundle)
        return false;

    if (!CFBundleLoadExecutable(bundle.get()))
        return false;

    m_bundle = WTFMove(bundle);
    return true;
}

void Module::unload()
{
    if (!m_bundle)
        return;

    // See the comment in Module.h for why we leak the bundle here.
    CFBundleRef unused = m_bundle.leakRef();
    (void)unused;
}

void* Module::platformFunctionPointer(const char* functionName) const
{
    if (!m_bundle)
        return 0;
    RetainPtr<CFStringRef> functionNameString = adoptCF(CFStringCreateWithCStringNoCopy(kCFAllocatorDefault, functionName, kCFStringEncodingASCII, kCFAllocatorNull));
    return CFBundleGetFunctionPointerForName(m_bundle.get(), functionNameString.get());
}

String Module::bundleIdentifier() const
{
    return CFBundleGetIdentifier(m_bundle.get());
}

}
