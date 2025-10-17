/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 20, 2023.
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

#include <wtf/Seconds.h>

#if PLATFORM(COCOA)
#include <wtf/RetainPtr.h>
OBJC_CLASS RSABSSATokenReady;
OBJC_CLASS RSABSSATokenWaitingActivation;
OBJC_CLASS RSABSSATokenBlinder;
#endif

namespace WebCore::PCM {

struct UnlinkableToken {
#if PLATFORM(COCOA)
    RetainPtr<RSABSSATokenBlinder> blinder;
    RetainPtr<RSABSSATokenWaitingActivation> waitingToken;
    RetainPtr<RSABSSATokenReady> readyToken;
#endif
    String valueBase64URL;
    
    UnlinkableToken isolatedCopy() const &;
    UnlinkableToken isolatedCopy() &&;
};

struct SourceUnlinkableToken : UnlinkableToken {
    SourceUnlinkableToken isolatedCopy() const & { return { UnlinkableToken::isolatedCopy() }; }
    SourceUnlinkableToken isolatedCopy() && { return { UnlinkableToken::isolatedCopy() }; }
};

struct DestinationUnlinkableToken : UnlinkableToken {
    DestinationUnlinkableToken isolatedCopy() const & { return { UnlinkableToken::isolatedCopy() }; }
    DestinationUnlinkableToken isolatedCopy() && { return { UnlinkableToken::isolatedCopy() }; }
};

struct SecretToken {
    String tokenBase64URL;
    String signatureBase64URL;
    String keyIDBase64URL;
    SecretToken isolatedCopy() const & { return { tokenBase64URL.isolatedCopy(), signatureBase64URL.isolatedCopy(), keyIDBase64URL.isolatedCopy() }; }
    SecretToken isolatedCopy() && { return { WTFMove(tokenBase64URL).isolatedCopy(), WTFMove(signatureBase64URL).isolatedCopy(), WTFMove(keyIDBase64URL).isolatedCopy() }; }
    bool isValid() const;
};

struct SourceSecretToken : SecretToken {
    SourceSecretToken isolatedCopy() const & { return { SecretToken::isolatedCopy() }; }
    SourceSecretToken isolatedCopy() && { return { SecretToken::isolatedCopy() }; }
};

struct DestinationSecretToken : SecretToken {
    DestinationSecretToken isolatedCopy() const & { return { SecretToken::isolatedCopy() }; }
    DestinationSecretToken isolatedCopy() && { return { SecretToken::isolatedCopy() }; }
};

}
