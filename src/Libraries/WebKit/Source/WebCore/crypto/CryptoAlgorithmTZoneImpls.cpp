/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 24, 2023.
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

#include "CryptoAlgorithmAesCbcCfbParams.h"
#include "CryptoAlgorithmAesCtrParams.h"
#include "CryptoAlgorithmAesGcmParams.h"
#include "CryptoAlgorithmAesKeyParams.h"
#include "CryptoAlgorithmEcKeyParams.h"
#include "CryptoAlgorithmEcdhKeyDeriveParams.h"
#include "CryptoAlgorithmEcdsaParams.h"
#include "CryptoAlgorithmHkdfParams.h"
#include "CryptoAlgorithmHmacKeyParams.h"
#include "CryptoAlgorithmParameters.h"
#include "CryptoAlgorithmPbkdf2Params.h"
#include "CryptoAlgorithmRsaHashedImportParams.h"
#include "CryptoAlgorithmRsaKeyGenParams.h"
#include "CryptoAlgorithmRsaOaepParams.h"
#include "CryptoAlgorithmRsaPssParams.h"
#include "CryptoAlgorithmX25519Params.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CryptoAlgorithmAesCbcCfbParams);
WTF_MAKE_TZONE_ALLOCATED_IMPL(CryptoAlgorithmAesCtrParams);
WTF_MAKE_TZONE_ALLOCATED_IMPL(CryptoAlgorithmAesGcmParams);
WTF_MAKE_TZONE_ALLOCATED_IMPL(CryptoAlgorithmAesKeyParams);
WTF_MAKE_TZONE_ALLOCATED_IMPL(CryptoAlgorithmEcKeyParams);
WTF_MAKE_TZONE_ALLOCATED_IMPL(CryptoAlgorithmEcdhKeyDeriveParams);
WTF_MAKE_TZONE_ALLOCATED_IMPL(CryptoAlgorithmEcdsaParams);
WTF_MAKE_TZONE_ALLOCATED_IMPL(CryptoAlgorithmHkdfParams);
WTF_MAKE_TZONE_ALLOCATED_IMPL(CryptoAlgorithmHmacKeyParams);
WTF_MAKE_TZONE_ALLOCATED_IMPL(CryptoAlgorithmPbkdf2Params);
WTF_MAKE_TZONE_ALLOCATED_IMPL(CryptoAlgorithmRsaHashedImportParams);
WTF_MAKE_TZONE_ALLOCATED_IMPL(CryptoAlgorithmRsaKeyGenParams);
WTF_MAKE_TZONE_ALLOCATED_IMPL(CryptoAlgorithmRsaOaepParams);
WTF_MAKE_TZONE_ALLOCATED_IMPL(CryptoAlgorithmRsaPssParams);
WTF_MAKE_TZONE_ALLOCATED_IMPL(CryptoAlgorithmX25519Params);
WTF_MAKE_TZONE_ALLOCATED_IMPL(CryptoAlgorithmParameters);

} // namespace WebCore
