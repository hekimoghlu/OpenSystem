/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 2, 2025.
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
#import "CDMFairPlayStreaming.h"

#if ENABLE(ENCRYPTED_MEDIA) && HAVE(AVCONTENTKEYSESSION)

#import "SharedBuffer.h"
#import <pal/spi/cocoa/AVFoundationSPI.h>
#import <wtf/Ref.h>
#import <wtf/Vector.h>
#import <wtf/cocoa/TypeCastsCocoa.h>

namespace WebCore {

Vector<Ref<SharedBuffer>> CDMPrivateFairPlayStreaming::keyIDsForRequest(AVContentKeyRequest *request)
{
    if (auto *identiferStr = dynamic_objc_cast<NSString>(request.identifier))
        return { SharedBuffer::create([identiferStr dataUsingEncoding:NSUTF8StringEncoding]) };
    if (auto *identifierData = dynamic_objc_cast<NSData>(request.identifier))
        return { SharedBuffer::create(identifierData) };
    if (request.initializationData) {
        if (auto sinfKeyIDs = CDMPrivateFairPlayStreaming::extractKeyIDsSinf(SharedBuffer::create(request.initializationData)))
            return WTFMove(sinfKeyIDs.value());
#if HAVE(FAIRPLAYSTREAMING_MTPS_INITDATA)
        if (auto mptsKeyIDs = CDMPrivateFairPlayStreaming::extractKeyIDsMpts(SharedBuffer::create(request.initializationData)))
            return WTFMove(mptsKeyIDs.value());
#endif
    }
    return { };
}

} // namespace WebCore

#endif // ENABLE(ENCRYPTED_MEDIA) && HAVE(AVCONTENTKEYSESSION)
