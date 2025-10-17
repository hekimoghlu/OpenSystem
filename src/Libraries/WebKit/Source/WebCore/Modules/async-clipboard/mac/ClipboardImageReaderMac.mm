/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 18, 2022.
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

#if PLATFORM(MAC)

#import "Document.h"
#import "SharedBuffer.h"
#import <wtf/cocoa/VectorCocoa.h>

namespace WebCore {

void ClipboardImageReader::readBuffer(const String&, const String&, Ref<SharedBuffer>&& buffer)
{
    if (m_mimeType == "image/png"_s) {
        auto image = adoptNS([[NSImage alloc] initWithData:buffer->createNSData().get()]);
        if (auto cgImage = [image CGImageForProposedRect:nil context:nil hints:nil]) {
            auto representation = adoptNS([[NSBitmapImageRep alloc] initWithCGImage:cgImage]);
            NSData* nsData = [representation representationUsingType:NSBitmapImageFileTypePNG properties:@{ }];
            m_result = Blob::create(m_document.get(), makeVector(nsData), m_mimeType);
        }
    }
}

} // namespace WebCore

#endif // PLATFORM(MAC)
