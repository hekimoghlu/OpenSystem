/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 6, 2023.
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
#include "CryptoAlgorithmRegistry.h"

#include "CryptoAlgorithmAESCBC.h"
#include "CryptoAlgorithmAESCTR.h"
#include "CryptoAlgorithmAESGCM.h"
#include "CryptoAlgorithmAESKW.h"
#include "CryptoAlgorithmECDH.h"
#include "CryptoAlgorithmECDSA.h"
#include "CryptoAlgorithmEd25519.h"
#include "CryptoAlgorithmHKDF.h"
#include "CryptoAlgorithmHMAC.h"
#include "CryptoAlgorithmPBKDF2.h"
#include "CryptoAlgorithmRSASSA_PKCS1_v1_5.h"
#include "CryptoAlgorithmRSA_OAEP.h"
#include "CryptoAlgorithmRSA_PSS.h"
#include "CryptoAlgorithmSHA1.h"
#include "CryptoAlgorithmSHA224.h"
#include "CryptoAlgorithmSHA256.h"
#include "CryptoAlgorithmSHA384.h"
#include "CryptoAlgorithmSHA512.h"
#include "CryptoAlgorithmX25519.h"

namespace WebCore {

void CryptoAlgorithmRegistry::platformRegisterAlgorithms()
{
    registerAlgorithm<CryptoAlgorithmAESCBC>();
    registerAlgorithm<CryptoAlgorithmAESCTR>();
    registerAlgorithm<CryptoAlgorithmAESGCM>();
    registerAlgorithm<CryptoAlgorithmAESKW>();
    registerAlgorithm<CryptoAlgorithmECDH>();
    registerAlgorithm<CryptoAlgorithmECDSA>();
    registerAlgorithm<CryptoAlgorithmEd25519>();
    registerAlgorithm<CryptoAlgorithmHKDF>();
    registerAlgorithm<CryptoAlgorithmHMAC>();
    registerAlgorithm<CryptoAlgorithmPBKDF2>();
    registerAlgorithm<CryptoAlgorithmRSASSA_PKCS1_v1_5>();
    registerAlgorithm<CryptoAlgorithmRSA_OAEP>();
    registerAlgorithm<CryptoAlgorithmRSA_PSS>();
    registerAlgorithm<CryptoAlgorithmSHA1>();
    registerAlgorithm<CryptoAlgorithmSHA224>();
    registerAlgorithm<CryptoAlgorithmSHA256>();
    registerAlgorithm<CryptoAlgorithmSHA384>();
    registerAlgorithm<CryptoAlgorithmSHA512>();
    registerAlgorithm<CryptoAlgorithmX25519>();
}

} // namespace WebCore
