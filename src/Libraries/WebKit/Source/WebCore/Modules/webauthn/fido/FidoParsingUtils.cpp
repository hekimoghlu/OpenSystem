/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 25, 2023.
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
#include "FidoParsingUtils.h"

#if ENABLE(WEB_AUTHN)

#include "FidoConstants.h"

namespace fido {

Vector<uint8_t> getInitPacketData(const Vector<uint8_t>& data)
{
    return data.subvector(0, std::min(kHidInitPacketDataSize, data.size()));
}

Vector<uint8_t> getContinuationPacketData(const Vector<uint8_t>& data, size_t beginPosition)
{
    if (beginPosition > data.size())
        return { };

    return data.subspan(beginPosition, std::min(kHidContinuationPacketDataSize, data.size() - beginPosition));
}

} // namespace fido

#endif // ENABLE(WEB_AUTHN)
