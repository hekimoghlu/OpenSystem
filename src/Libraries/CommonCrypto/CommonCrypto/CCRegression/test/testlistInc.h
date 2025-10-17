/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 26, 2022.
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
// Random MUST BE FIRST
ONE_TEST(CommonRandom)

// No particular sequence
ONE_TEST(CommonCPP)
ONE_TEST(CommonCryptoSymECB)
ONE_TEST(CommonCryptoSymCBC)
ONE_TEST(CommonCryptoSymOFB)
ONE_TEST(CommonCryptoSymGCM)
ONE_TEST(CommonCryptoSymCCM)
ONE_TEST(CommonCryptoSymCTR)
ONE_TEST(CommonCryptoSymXTS)
ONE_TEST(CommonCryptoSymRC2)
ONE_TEST(CommonCryptoSymRegression)
ONE_TEST(CommonCryptoSymOffset)
ONE_TEST(CommonCryptoSymZeroLength)
ONE_TEST(CommonCryptoOutputLength)
ONE_TEST(CommonCryptoNoPad)
ONE_TEST(CommonCryptoSymCFB)
ONE_TEST(CommonCryptoCTSPadding)
ONE_TEST(CommonSymmetricWrap)
ONE_TEST(CommonDH)
ONE_TEST(CommonDigest)
ONE_TEST(CommonHMac)
ONE_TEST(CommonCryptoReset)
ONE_TEST(CommonCryptoSymChaCha20)
ONE_TEST(CommonCryptoSymChaCha20Poly1305)
#if !defined(_WIN32)
ONE_TEST(CommonBigNum) /* BignNm is not ported to Windows */
#endif
ONE_TEST(CommonBigDigest)
ONE_TEST(CommonCryptoWithData)
ONE_TEST(CommonCryptoBlowfish)
ONE_TEST(CommonCRCTest)
ONE_TEST(CommonBaseEncoding)
ONE_TEST(CommonHKDF)
ONE_TEST(CommonANSIKDF)
ONE_TEST(CommonNISTKDF)
ONE_TEST(CommonKeyDerivation)
ONE_TEST(CommonEC)
ONE_TEST(CommonRSA)
ONE_TEST(CommonHMacClone)
ONE_TEST(CommonCMac)
ONE_TEST(CommonCollabKeyGen)

