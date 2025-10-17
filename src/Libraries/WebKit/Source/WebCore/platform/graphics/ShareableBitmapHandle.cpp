/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 15, 2021.
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
#include "ShareableBitmapHandle.h"

namespace WebCore {

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER_AND_EXPORT(ShareableBitmapHandle, WTF_INTERNAL);
DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(ShareableBitmapHandle);

ShareableBitmapHandle::ShareableBitmapHandle(SharedMemory::Handle&& handle, const ShareableBitmapConfiguration& config)
    : m_handle(WTFMove(handle))
    , m_configuration(config)
{
}

void ShareableBitmapHandle::takeOwnershipOfMemory(MemoryLedger ledger) const
{
    m_handle.takeOwnershipOfMemory(ledger);
}

void ShareableBitmapHandle::setOwnershipOfMemory(const ProcessIdentity& identity, MemoryLedger ledger) const
{
    m_handle.setOwnershipOfMemory(identity, ledger);
}


} // namespace WebCore
