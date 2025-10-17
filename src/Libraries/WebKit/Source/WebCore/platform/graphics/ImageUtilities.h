/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 27, 2024.
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

#include "IntSize.h"

#include <wtf/Forward.h>
#include <wtf/WorkQueue.h>

namespace WebCore {

class ShareableBitmap;
class SharedBuffer;

WEBCORE_EXPORT WorkQueue& sharedImageTranscodingQueue();

// Given a list of files' 'paths' and 'allowedMIMETypes', the function returns a list
// of strings whose size is the same as the size of 'paths' and its entries are all
// null strings except the ones whose MIME types are not in 'allowedMIMETypes'.
WEBCORE_EXPORT Vector<String> findImagesForTranscoding(const Vector<String>& paths, const Vector<String>& allowedMIMETypes);

// Given a list of images' 'paths', this function transcodes these images to a new
// format whose UTI is destinationUTI. The result of the transcoding will be written
// to temporary files whose extensions are 'destinationExtension'. It returns a list
// of paths to the result temporary files. If an entry in 'paths' is null or an error
// happens while transcoding, a null string will be added to the returned list.
WEBCORE_EXPORT Vector<String> transcodeImages(const Vector<String>& paths, const String& destinationUTI, const String& destinationExtension);

enum class ImageDecodingError : uint8_t {
    Internal,
    BadData,
    UnsupportedType
};
WEBCORE_EXPORT String descriptionString(ImageDecodingError);
WEBCORE_EXPORT Expected<std::pair<String, Vector<IntSize>>, ImageDecodingError> utiAndAvailableSizesFromImageData(std::span<const uint8_t>);
WEBCORE_EXPORT void createBitmapsFromImageData(std::span<const uint8_t> data, std::span<const unsigned> lengths, CompletionHandler<void(Vector<Ref<ShareableBitmap>>&&)>&&);
WEBCORE_EXPORT RefPtr<SharedBuffer> createIconDataFromBitmaps(Vector<Ref<ShareableBitmap>>&&);
WEBCORE_EXPORT void decodeImageWithSize(std::span<const uint8_t> data, std::optional<FloatSize>, CompletionHandler<void(RefPtr<ShareableBitmap>&&)>&&);

} // namespace WebCore

