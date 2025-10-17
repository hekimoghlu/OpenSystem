/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 17, 2022.
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
#include "AppHighlightRangeData.h"

#if ENABLE(APP_HIGHLIGHTS)

#include "Document.h"
#include "DocumentMarkerController.h"
#include "HTMLBodyElement.h"
#include "Logging.h"
#include "Node.h"
#include "RenderedDocumentMarker.h"
#include "SharedBuffer.h"
#include "SimpleRange.h"
#include "StaticRange.h"
#include "TextIterator.h"
#include "WebCorePersistentCoders.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/persistence/PersistentCoders.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(AppHighlightRangeData);

std::optional<AppHighlightRangeData> AppHighlightRangeData::create(const FragmentedSharedBuffer& buffer)
{
    auto contiguousBuffer = buffer.makeContiguous();
    auto decoder = contiguousBuffer->decoder();
    std::optional<AppHighlightRangeData> data;
    decoder >> data;
    return data;
}

Ref<FragmentedSharedBuffer> AppHighlightRangeData::toSharedBuffer() const
{
    WTF::Persistence::Encoder encoder;
    encoder << *this;
    return SharedBuffer::create(encoder.span());
}

} // namespace WebCore

#endif
