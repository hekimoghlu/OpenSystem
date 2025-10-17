/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 15, 2023.
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

#include "BlobPart.h"
#include "EndingType.h"

namespace JSC {
class ArrayBuffer;
class ArrayBufferView;
}

namespace WebCore {

class Blob;

class BlobBuilder {
public:
    BlobBuilder(EndingType);

    void append(RefPtr<JSC::ArrayBuffer>&&);
    void append(RefPtr<JSC::ArrayBufferView>&&);
    void append(RefPtr<Blob>&&);
    void append(const String& text);

    Vector<BlobPart> finalize();

private:
    EndingType m_endings;
    Vector<BlobPart> m_items;
    Vector<uint8_t> m_appendableData;
};

} // namespace WebCore
