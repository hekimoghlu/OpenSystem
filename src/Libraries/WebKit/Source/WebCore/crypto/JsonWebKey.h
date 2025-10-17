/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 26, 2022.
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

#include "CryptoKeyUsage.h"
#include "RsaOtherPrimesInfo.h"
#include <wtf/Vector.h>

namespace WebCore {

struct JsonWebKey {
    JsonWebKey isolatedCopy() && {
        return {
            crossThreadCopy(WTFMove(kty)),
            crossThreadCopy(WTFMove(use)),
            key_ops,
            usages,
            crossThreadCopy(WTFMove(alg)),
            ext,
            crossThreadCopy(WTFMove(crv)),
            crossThreadCopy(WTFMove(x)),
            crossThreadCopy(WTFMove(y)),
            crossThreadCopy(WTFMove(d)),
            crossThreadCopy(WTFMove(n)),
            crossThreadCopy(WTFMove(e)),
            crossThreadCopy(WTFMove(p)),
            crossThreadCopy(WTFMove(q)),
            crossThreadCopy(WTFMove(dp)),
            crossThreadCopy(WTFMove(dq)),
            crossThreadCopy(WTFMove(qi)),
            crossThreadCopy(WTFMove(oth)),
            crossThreadCopy(WTFMove(k))
        };
    }

    String kty;
    String use;
    // FIXME: Consider merging key_ops and usages.
    std::optional<Vector<CryptoKeyUsage>> key_ops;
    CryptoKeyUsageBitmap usages;
    String alg;

    std::optional<bool> ext;

    String crv;
    String x;
    String y;
    String d;
    String n;
    String e;
    String p;
    String q;
    String dp;
    String dq;
    String qi;
    std::optional<Vector<RsaOtherPrimesInfo>> oth;
    String k;
};

} // namespace WebCore
