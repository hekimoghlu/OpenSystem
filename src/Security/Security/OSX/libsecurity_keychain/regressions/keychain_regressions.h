/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 1, 2023.
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

ONE_TEST(kc_01_keychain_creation)
ONE_TEST(kc_01_corrupt_keychain)
ONE_TEST(kc_02_unlock_noui)
ONE_TEST(kc_03_status)
ONE_TEST(kc_03_keychain_list)
ONE_TEST(kc_04_is_valid)
ONE_TEST(kc_05_find_existing_items)
ONE_TEST(kc_05_find_existing_items_locked)
ONE_TEST(kc_06_cert_search_email)
ONE_TEST(kc_10_item_add_generic)
ONE_TEST(kc_10_item_add_internet)
ONE_TEST(kc_10_item_add_certificate)
ONE_TEST(kc_12_key_create_symmetric)
ONE_TEST(kc_12_key_create_symmetric_and_use)
ONE_TEST(kc_15_key_update_valueref)
ONE_TEST(kc_15_item_update_label_skimaad)
ONE_TEST(kc_16_item_update_password)
ONE_TEST(kc_17_item_find_key)
ONE_TEST(kc_18_find_combined)
ONE_TEST(kc_19_item_copy_internet)
ONE_TEST(kc_20_identity_persistent_refs)
ONE_TEST(kc_20_identity_key_attributes)
ONE_TEST(kc_20_identity_find_stress)
ONE_TEST(kc_20_key_find_stress)
ONE_TEST(kc_20_item_add_stress)
ONE_TEST(kc_20_item_find_stress)
ONE_TEST(kc_20_item_delete_stress)
ONE_TEST(kc_21_item_use_callback)
ONE_TEST(kc_21_item_xattrs)
ONE_TEST(kc_23_key_export_symmetric)
ONE_TEST(kc_24_key_copy_keychain)
ONE_TEST(kc_26_key_import_public)
ONE_TEST(kc_27_key_non_extractable)
ONE_TEST(kc_28_p12_import)
ONE_TEST(kc_28_cert_sign)
ONE_TEST(kc_30_xara)
ONE_TEST(kc_40_seckey)
ONE_TEST(kc_41_sececkey)
ONE_TEST(kc_43_seckey_interop)
ONE_TEST(kc_44_secrecoverypassword)
ONE_TEST(kc_45_change_password)
ONE_TEST(si_20_certificate_copy_values)
ONE_TEST(si_33_keychain_backup)
ONE_TEST(si_34_one_true_keychain)
ONE_TEST(si_40_identity_tests)
