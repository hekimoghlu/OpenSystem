/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 24, 2024.
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

#if USE(CG)

#include <optional>
#include <wtf/Forward.h>

namespace WebCore {

class PixelBuffer;

WEBCORE_EXPORT uint8_t verifyImageBufferIsBigEnough(std::span<const uint8_t> buffer);

RetainPtr<CFStringRef> utiFromImageBufferMIMEType(const String& mimeType);
CFStringRef jpegUTI();
Vector<uint8_t> encodeData(CGImageRef, const String& mimeType, std::optional<double> quality);
Vector<uint8_t> encodeData(const PixelBuffer&, const String& mimeType, std::optional<double> quality);
Vector<uint8_t> encodeData(std::span<const uint8_t>, const String& mimeType, std::optional<double> quality);

WEBCORE_EXPORT String dataURL(CGImageRef, const String& mimeType, std::optional<double> quality);
String dataURL(const PixelBuffer&, const String& mimeType, std::optional<double> quality);

} // namespace WebCore

#endif // USE(CG)
