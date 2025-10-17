/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 22, 2025.
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

#include "FormData.h"
#include <wtf/Forward.h>

namespace PAL {
class TextEncoding;
}

namespace WebCore {

namespace FormDataBuilder {

// Helper functions used by HTMLFormElement for multi-part form data.
Vector<uint8_t> generateUniqueBoundaryString();
void beginMultiPartHeader(Vector<uint8_t>&, std::span<const uint8_t> boundary, const Vector<uint8_t>& name);
void addBoundaryToMultiPartHeader(Vector<uint8_t>&, std::span<const uint8_t> boundary, bool isLastBoundary = false);
void addFilenameToMultiPartHeader(Vector<uint8_t>&, const PAL::TextEncoding&, const String& filename);
void addContentTypeToMultiPartHeader(Vector<uint8_t>&, const CString& mimeType);
void finishMultiPartHeader(Vector<uint8_t>&);

// Helper functions used by HTMLFormElement for non-multi-part form data.
void addKeyValuePairAsFormData(Vector<uint8_t>&, const Vector<uint8_t>& key, const Vector<uint8_t>& value, FormData::EncodingType = FormData::EncodingType::FormURLEncoded);
void encodeStringAsFormData(Vector<uint8_t>&, const CString&);

}

}
