/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 9, 2023.
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
#pragma once

#if ENABLE(PICTURE_IN_PICTURE_API)

#include "HTMLVideoElement.h"
#include "Supplementable.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class DeferredPromise;
class Document;

class DocumentPictureInPicture : public Supplement<Document> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(DocumentPictureInPicture);
public:
    virtual ~DocumentPictureInPicture();

    static bool pictureInPictureEnabled(Document&);
    static void exitPictureInPicture(Document&, Ref<DeferredPromise>&&);

    static DocumentPictureInPicture* from(Document&);

private:
    static ASCIILiteral supplementName() { return "DocumentPictureInPicture"_s; };
};

} // namespace WebCore

#endif // ENABLE(PICTURE_IN_PICTURE_API)
