/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 2, 2025.
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
#include <regressions/test/testmore.h>

ONE_TEST(si_25_cms_skid)
ONE_TEST(si_26_cms_apple_signed_samples)
ONE_TEST(si_27_cms_parse)
ONE_TEST(si_29_cms_chain_mode)
ONE_TEST(si_34_cms_timestamp)
ONE_TEST(si_35_cms_expiration_time)
ONE_TEST(si_41_sececkey)
ONE_TEST(si_44_seckey_gen)
ONE_TEST(si_44_seckey_rsa)
ONE_TEST(si_44_seckey_ec)
ONE_TEST(si_44_seckey_ies)
ONE_TEST(si_44_seckey_aks)
#if TARGET_OS_IOS && !TARGET_OS_SIMULATOR
ONE_TEST(si_44_seckey_fv)
#endif
ONE_TEST(si_44_seckey_proxy)
ONE_TEST(si_60_cms)
ONE_TEST(si_61_pkcs12)
ONE_TEST(si_62_csr)
ONE_TEST(si_63_scep)
ONE_TEST(si_64_ossl_cms)
ONE_TEST(si_65_cms_cert_policy)
ONE_TEST(si_66_smime)
ONE_TEST(si_68_secmatchissuer)
ONE_TEST(si_69_keydesc)
ONE_TEST(si_89_cms_hash_agility)
ONE_TEST(si_96_csr_acme)
ONE_TEST(rk_01_recoverykey)

ONE_TEST(padding_00_mmcs)
