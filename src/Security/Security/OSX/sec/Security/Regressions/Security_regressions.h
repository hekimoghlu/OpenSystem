/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 25, 2024.
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

ONE_TEST(pbkdf2_00_hmac_sha1)
ONE_TEST(spbkdf_00_hmac_sha1)
ONE_TEST(spbkdf_01_hmac_sha256)

ONE_TEST(si_00_find_nothing)
ONE_TEST(si_05_add)
ONE_TEST(si_10_find_internet)
ONE_TEST(si_11_update_data)
ONE_TEST(si_12_item_stress)
ONE_TEST(si_14_dateparse)
DISABLED_ONE_TEST(si_15_delete_access_group)
ONE_TEST(si_17_item_system_bluetooth)
DISABLED_ONE_TEST(si_30_keychain_upgrade) //obsolete, needs updating
DISABLED_ONE_TEST(si_31_keychain_bad)
DISABLED_ONE_TEST(si_31_keychain_unreadable)
#if !TARGET_OS_TV && !TARGET_OS_WATCH && !TARGET_OS_XR
ONE_TEST(si_33_keychain_backup)
#endif
ONE_TEST(si_40_seckey)
ONE_TEST(si_40_seckey_custom)
ONE_TEST(si_42_identity)
ONE_TEST(si_43_persistent)
ONE_TEST(si_50_secrandom)
ONE_TEST(si_72_syncableitems)
ONE_TEST(si_73_secpasswordgenerate)
#if TARGET_OS_IPHONE
#if TARGET_OS_SIMULATOR
OFF_ONE_TEST(si_76_shared_credentials)
#else
ONE_TEST(si_76_shared_credentials)
#endif
ONE_TEST(si_77_SecAccessControl)
#else
DISABLED_ONE_TEST(si_76_shared_credentials)
DISABLED_ONE_TEST(si_77_SecAccessControl)
#endif
ONE_TEST(si_78_query_attrs)
ONE_TEST(si_80_empty_data)
ONE_TEST(si_82_token_ag)
ONE_TEST(si_90_emcs)
ONE_TEST(si_95_cms_basic)

ONE_TEST(otr_00_identity)
ONE_TEST(otr_30_negotiation)
ONE_TEST(otr_otrdh)
ONE_TEST(otr_packetdata)
ONE_TEST(otr_40_edgecases)
ONE_TEST(otr_50_roll)
ONE_TEST(otr_60_slowroll)

#if TARGET_OS_IPHONE
ONE_TEST(so_01_serverencryption)
#else
DISABLED_ONE_TEST(so_01_serverencryption)
#endif
