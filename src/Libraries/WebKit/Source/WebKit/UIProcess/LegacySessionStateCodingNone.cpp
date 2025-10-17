/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 3, 2024.
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
#include "LegacySessionStateCoding.h"

#include "APIData.h"
#include "ArgumentCoders.h"
#include "Decoder.h"
#include "Encoder.h"
#include "MessageNames.h"
#include "SessionState.h"

namespace WebKit {

RefPtr<API::Data> encodeLegacySessionState(const SessionState& sessionState)
{
    // FIXME: This should use WTF::Persistence::Encoder instead.
    IPC::Encoder encoder(IPC::MessageName::LegacySessionState, 0);
    encoder << sessionState.backForwardListState;
    encoder << sessionState.renderTreeSize;
    encoder << sessionState.provisionalURL;
    return API::Data::create(encoder.span());
}

bool decodeLegacySessionState(std::span<const uint8_t> data, SessionState& sessionState)
{
    auto decoder = IPC::Decoder::create(data, { });
    if (!decoder)
        return false;

    std::optional<BackForwardListState> backForwardListState;
    *decoder >> backForwardListState;
    if (!backForwardListState)
        return false;
    sessionState.backForwardListState = WTFMove(*backForwardListState);

    std::optional<uint64_t> renderTreeSize;
    *decoder >> renderTreeSize;
    if (!renderTreeSize)
        return false;
    sessionState.renderTreeSize = *renderTreeSize;

    std::optional<URL> provisionalURL;
    *decoder >> provisionalURL;
    if (!provisionalURL)
        return false;
    sessionState.provisionalURL = WTFMove(*provisionalURL);

    return true;
}

} // namespace WebKit
