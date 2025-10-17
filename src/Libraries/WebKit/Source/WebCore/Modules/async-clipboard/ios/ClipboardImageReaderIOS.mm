/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 30, 2025.
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
#import "config.h"
#import "ClipboardImageReader.h"

#if PLATFORM(IOS_FAMILY)

#import "Document.h"
#import "SharedBuffer.h"
#import <wtf/cocoa/VectorCocoa.h>

#import <pal/ios/UIKitSoftLink.h>

namespace WebCore {

void ClipboardImageReader::readBuffer(const String&, const String&, Ref<SharedBuffer>&& buffer)
{
    if (m_mimeType == "image/png"_s) {
        auto image = adoptNS([PAL::allocUIImageInstance() initWithData:buffer->createNSData().get()]);
        if (auto nsData = UIImagePNGRepresentation(image.get()))
            m_result = Blob::create(m_document.get(), makeVector(nsData), m_mimeType);
    }
}

} // namespace WebCore

#endif // PLATFORM(IOS_FAMILY)
