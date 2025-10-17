/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 31, 2022.
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
#include "pas_config.h"

#if LIBPAS_ENABLED

#include "pas_race_test_hooks.h"

#if PAS_ENABLE_TESTING

pas_race_test_hook_callback pas_race_test_hook_callback_instance = NULL;
pas_race_test_lock_callback pas_race_test_will_lock_callback = NULL;
pas_race_test_lock_callback pas_race_test_did_lock_callback = NULL;
pas_race_test_lock_callback pas_race_test_did_try_lock_callback = NULL;
pas_race_test_lock_callback pas_race_test_will_unlock_callback = NULL;

#endif /* PAS_ENABLE_TESTING */

#endif /* LIBPAS_ENABLED */
